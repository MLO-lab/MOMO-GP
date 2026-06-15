from matplotlib import pyplot as plt
import os
import numpy as np
import sys
from sklearn.metrics import *
data_folder_path = '../../../data'
sys.path.append(data_folder_path)
import tensorflow as tf
import gpflow
import gpflux
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras import layers


def get_free_gpu_idx():
    """Get the index of the GPU with current lowest memory usage."""
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return np.argmin(memory_available)

gpu_idx = get_free_gpu_idx()
print("Using GPU #%s" % gpu_idx)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)


class GPD():
    def __init__(
        self,
        I,
        J,
        K,
        M1,
        M2,
        emb_sizes,
        batch_size=512,
        obs_mean1=None,
        obs_mean2=None,
        emb_reg=1e-3,
        lr=1e-3,
        save_path="./",
        likelihood="gaussian",
    ):
        """
        :param I: integer, number of entities in the first dimension.
        :param J: integer, number of entities in the second dimension.
        :param K: integer, number of entities in the third dimension.
        :param M: integer, number of inducing points.
        :param emb_sizes: a list of embedding sizes as integers.
        :param batch_size: integer, mini batch size for training, necessary to define the placeholders.
        :param obs_mean: integer, mean of training targets, optional.
        :param emb_reg: float, regularization term for embeddings.
        :param lr: float, learning rate.
        """

        self.I = I
        self.J = J
        self.K = K
        self.M1 = M1
        self.M2 = M2
        self.emb_sizes = emb_sizes
        self.batch_size = batch_size
        self.obs_mean1 = obs_mean1
        self.obs_mean2 = obs_mean2
        self.emb_reg = emb_reg
        self.lr = lr
        self.save_path = save_path
        self.emb1 = None
        self.emb2 = None
        self.emb3 = None
        self.kernel1 = None
        self.kernel2 = None
        self.model = None
        self.Z_size1 = None
        self.Z_size2 = None
        # num_data for each view's GPLayer = size of the triple store actually fed to
        # fit(). Set in train(); falls back to the dense I*J / I*K in make_model(). This
        # MUST match the fed store size so the ELBO's KL term is scaled correctly under
        # negative subsampling (where the store is smaller than I*J).
        self.num_data1 = None
        self.num_data2 = None
        # Observation likelihood: "gaussian" (default, z-scored targets), "nb" (raw counts),
        # "bernoulli" (binary, e.g. ATAC), or a per-view tuple e.g. ("nb", "bernoulli").
        self.likelihood_spec = likelihood
        # Populated by make_model(); reused by the GradientTape trainer.
        self.gp_layer1 = None
        self.gp_layer2 = None
        self.likelihood = self.likelihood1 = self.likelihood2 = None
        self.loss_fn = self.loss_fn1 = self.loss_fn2 = None


    def make_kernels(self, emb_size, kernels=None, active_dims=None):
        """
        :param emb_size: integer, embedding size for a kernel of one specific dimension.
        :param kernels: a list of strings, e.g. ['RBF', 'White], the sum of which shall form the kernel for one
        specific dimension.
        :param active_dims: active dimension of the kernel.
        :return: one kernel being the sum of all kernels required by the parameter 'kernels'.
        """ 
        kern = None
        if "RBF" in kernels:
            kern = gpflow.kernels.RBF(active_dims=active_dims, lengthscales=np.ones(emb_size))
            if "White" in kernels:
                kern = kern + gpflow.kernels.White()

        if "Linear" in kernels:
            kern = gpflow.kernels.Linear(active_dims=active_dims)  # lengthscales=np.ones(emb_size))
            if "White" in kernels:
                kern = kern + gpflow.kernels.White()

        return kern


    def build(self, kernels=["RBF"], first_pcs=None, second_pcs=None, third_pcs=None):  # , **kwargs    
        """
        Building the GP-Decomposition model, partially by calling build_svgp or build_sgpr.
        :param kernels: list of strings, defining the kernel structure for each embedding.
        :param optimiser: string, currently either 'adam' or 'adagrad'.
        :return: None.
        """
        # Embedding dtype follows gpflow's configured float so the whole model is consistent.
        # Default (float64) -> tf.float64 (unchanged). Set gpflow.config.set_default_float
        # (np.float32) BEFORE build() to halve memory (see run_full.py --float32).
        emb_dtype = tf.as_dtype(gpflow.default_float())

        if first_pcs is None:
            self.emb1 = tf.keras.Sequential(tf.keras.layers.Embedding(
                input_dim=self.I+1,
                output_dim=self.emb_sizes[0],
                dtype=emb_dtype,
                embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
                name="emb1",
                mask_zero=True
                
            ))
        else:
            self.emb1 = tf.keras.Sequential(tf.keras.layers.Embedding(
                input_dim=self.I+1,
                output_dim=self.emb_sizes[0],
                dtype=emb_dtype,
                embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
                name="emb1",
                mask_zero=True,
                embeddings_initializer = tf.keras.initializers.Constant(value=first_pcs)
            ))
            
        if second_pcs is None:    
            self.emb2 = tf.keras.Sequential(tf.keras.layers.Embedding(
                input_dim=self.J+1,
                output_dim=self.emb_sizes[1],
                dtype=emb_dtype,
                embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
                name="emb2",
                mask_zero=True
            ))
        else:
            self.emb2 = tf.keras.Sequential(tf.keras.layers.Embedding(
                input_dim=self.J+1,
                output_dim=self.emb_sizes[1],
                dtype=emb_dtype,
                embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
                name="emb2",
                mask_zero=True,
                embeddings_initializer = tf.keras.initializers.Constant(value=second_pcs)
            ))
        self.emb1.build()
        self.emb2.build()
        
        if self.K is not None:
            if third_pcs is None:
                self.emb3 = tf.keras.Sequential(tf.keras.layers.Embedding(
                    input_dim=self.K+1,
                    output_dim=self.emb_sizes[2],
                    dtype=emb_dtype,
                    embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
                    name="emb3",
                    mask_zero=True
                ))
            else:
                self.emb3 = tf.keras.Sequential(tf.keras.layers.Embedding(
                    input_dim=self.K+1,
                    output_dim=self.emb_sizes[2],
                    dtype=emb_dtype,
                    embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
                    name="emb3",
                    mask_zero=True,
                    embeddings_initializer = tf.keras.initializers.Constant(value=third_pcs)
                ))
            self.emb3.build()
            

        kernels1_1 = self.make_kernels(
            emb_size=self.emb_sizes[0],
            kernels=kernels,
            active_dims=np.arange(0, self.emb_sizes[0]),
        )
        
        kernels1_2 = self.make_kernels(
            emb_size=self.emb_sizes[0],
            kernels=kernels,
            active_dims=np.arange(0, self.emb_sizes[0]),
        )

        kernels2 = self.make_kernels(
            emb_size=self.emb_sizes[1],
            kernels=kernels,
            active_dims=np.arange(
                self.emb_sizes[0], self.emb_sizes[0] + self.emb_sizes[1]
            ),
        )
    
        self.kernel1 = kernels1_1 * kernels2
        if self.K is not None:
            kernels3 = self.make_kernels(
                emb_size=self.emb_sizes[2],
                kernels=kernels,
                active_dims=np.arange(
                    self.emb_sizes[0], self.emb_sizes[0] + self.emb_sizes[2]
                ),
            )
            self.kernel2 = kernels1_2 * kernels3
            
        if self.obs_mean1 is not None:
            observations_mean = tf.constant([self.obs_mean1], dtype=tf.float64)
            self.mean_fn1 = lambda _: observations_mean[:, None]
        else:
            self.mean_fn1 = self.obs_mean1
            
        if self.obs_mean2 is not None:
            observations_mean = tf.constant([self.obs_mean2], dtype=tf.float64)
            self.mean_fn2 = lambda _: observations_mean[:, None]
        else:
            self.mean_fn2 = self.obs_mean2   

        self.Z_size1 = self.emb_sizes[0] + self.emb_sizes[1]
        if self.K is not None:
            self.Z_size2 = self.emb_sizes[0] + self.emb_sizes[2]

    def set_inducing_points(self, X_tr1, X_tr2=None):
        ind = np.random.choice(range(X_tr1.shape[0]), self.M1, replace=False)
        inducing_variable = X_tr1[ind, :]
        inducing_variable_emb1 = self.emb1(inducing_variable[:, 0])
        inducing_variable_emb2 = self.emb2(inducing_variable[:, 1])
        hZ1 = tf.concat([inducing_variable_emb1, inducing_variable_emb2], axis=1)
        self.Z_0_1 = gpflow.inducing_variables.InducingPoints(hZ1)
        if self.K is not None:
            ind = np.random.choice(range(X_tr2.shape[0]), self.M2, replace=False)
            inducing_variable = X_tr2[ind, :]
            inducing_variable_emb1 = self.emb1(inducing_variable[:, 0])
            inducing_variable_emb3 = self.emb3(inducing_variable[:, 1])
            hZ2 = tf.concat([inducing_variable_emb1, inducing_variable_emb3], axis=1)  
            self.Z_0_2 = gpflow.inducing_variables.InducingPoints(hZ2)
    
    
    def make_model(self):
        def myMask(x):
            import keras.backend as Kbackend
            mask= Kbackend.greater(x,0) #will return boolean values
            mask= Kbackend.cast(mask, dtype=Kbackend.floatx()) 
            return mask
    
        from .likelihoods import make_likelihood
        # Resolve the likelihood spec: a single name (both views) or (name1, name2).
        spec = self.likelihood_spec
        if isinstance(spec, (list, tuple)):
            name1, name2 = spec[0], spec[1]
            single_shared = False
        else:
            name1 = name2 = spec
            single_shared = True  # legacy: one shared likelihood object across both views

        lik1, n1 = make_likelihood(name1)
        # num_latent_gps: count/binary heads use 1 latent; Gaussian keeps the legacy r1+rf.
        nlat1 = n1 if n1 is not None else self.emb_sizes[0] + self.emb_sizes[1]
        container1 = gpflux.layers.TrackableLayer()
        container1.likelihood = lik1
        loss1 = gpflux.losses.LikelihoodLoss(lik1)
        # Stored so the custom GradientTape trainer reuses the exact likelihood / loss.
        self.likelihood = self.likelihood1 = lik1
        self.loss_fn = self.loss_fn1 = loss1

        inputA1 = Input(shape=1)
        xA = self.emb1(inputA1)
        tmp = tf.keras.layers.Lambda(lambda val: myMask(val))(inputA1)
        xA = Multiply(name = 'masked_embedding_cells1')([xA,tmp])

        inputB1 = Input(shape=1)
        xB = self.emb2(inputB1)
        tmp = tf.keras.layers.Lambda(lambda val: myMask(val))(inputB1)
        xB = Multiply(name = 'masked_embedding_genes')([xB,tmp])

        modelA1 = Model(inputs=inputA1, outputs=xA)
        modelB1 = Model(inputs=inputB1, outputs=xB)
        combined = tf.reshape(tf.keras.layers.concatenate([modelA1.output, modelB1.output]),[-1, self.emb_sizes[0]+self.emb_sizes[1]])
        num_data1 = self.num_data1 if self.num_data1 is not None else self.I*self.J
        # GPLayer's default mean_function is Identity (maps the r1+r2-dim input through),
        # which is why the Gaussian head uses num_latent_gps=r1+r2. A 1-latent count/binary
        # head needs a Zero mean so the GP output is genuinely scalar (log-rate / logit).
        mean_fn1 = gpflow.mean_functions.Zero() if n1 is not None else None
        gp_layer1 = gpflux.layers.GPLayer(self.kernel1, self.Z_0_1, num_data=num_data1, num_latent_gps=nlat1, mean_function=mean_fn1)
        xC = gp_layer1(combined)
        Output1 = container1(xC)
        self.gp_layer1 = gp_layer1

        if self.K is not None:
            # View-2 likelihood: shared with view 1 for a single-string spec (legacy), else
            # its own (e.g. 'nb' for RNA + 'bernoulli' for ATAC).
            if single_shared:
                lik2, loss2, container2, n2 = lik1, loss1, container1, n1
            else:
                lik2, n2 = make_likelihood(name2)
                loss2 = gpflux.losses.LikelihoodLoss(lik2)
                container2 = gpflux.layers.TrackableLayer()
                container2.likelihood = lik2
            nlat2 = n2 if n2 is not None else self.emb_sizes[0] + self.emb_sizes[2]
            self.likelihood2 = lik2
            self.loss_fn2 = loss2

            inputA2 = Input(shape=1)
            xA = self.emb1(inputA2)
            tmp = tf.keras.layers.Lambda(lambda val: myMask(val))(inputA2)
            xA = Multiply(name = 'masked_embedding_cells2')([xA,tmp])

            inputB2 = Input(shape=1)
            xB = self.emb3(inputB2)
            tmp = tf.keras.layers.Lambda(lambda val: myMask(val))(inputB2)
            xB = Multiply(name = 'masked_embedding_peaks')([xB,tmp])

            modelA2 = Model(inputs=inputA2, outputs=xA)
            modelB2 = Model(inputs=inputB2, outputs=xB)
            combined = tf.reshape(tf.keras.layers.concatenate([modelA2.output, modelB2.output]),[-1, self.emb_sizes[0]+self.emb_sizes[2]])
            num_data2 = self.num_data2 if self.num_data2 is not None else self.I*self.K
            mean_fn2 = gpflow.mean_functions.Zero() if n2 is not None else None
            gp_layer2 = gpflux.layers.GPLayer(self.kernel2, self.Z_0_2, num_data=num_data2, num_latent_gps=nlat2, mean_function=mean_fn2)
            xC = gp_layer2(combined)
            Output2 = container2(xC)
            self.gp_layer2 = gp_layer2
            self.model = Model(inputs=[inputA1,inputB1,inputA2,inputB2], outputs=[Output1,Output2])
            self.model.compile(loss=[loss1, loss2], optimizer="adam")
        else:
            self.model = Model(inputs=[inputA1,inputB1], outputs=Output1)
            self.model.compile(loss=loss1, optimizer="adam")
              
    def _train_tape(self, X_tr1, Y_tr1, X_tr2, Y_tr2, sample_weight1, sample_weight2, n_iter):
        """Custom GradientTape loop batching the two views INDEPENDENTLY.

        Keras ``fit`` forces a shared batch axis, so both views process ``~max(n1, n2)``
        rows/epoch (padding or cycling). Here each view gets a batch proportional to its
        own size, so total work per epoch is ``~n1 + n2`` — the genuine compute win for
        lopsided feature counts (e.g. 2k genes vs 100k ATAC peaks).

        The objective is assembled from the SAME primitives the Keras model uses, so it is
        equivalent: ``sum_views[ weighted_mean_data_loss + (prior_kl - log_prior)/num_data ]
        + emb_reg * ||embeddings||^2`` (the GPLayer KL formula is replicated exactly).

        NB: unlike ``fit`` (which hard-codes ``optimizer="adam"`` and so ignores ``lr``),
        this path uses ``Adam(self.lr)`` — it actually honours the configured learning rate.
        """
        n1, n2 = len(Y_tr1), len(Y_tr2)
        steps = int(np.ceil(max(n1, n2) / self.batch_size))
        # Per-view batch sizes: each view covers its data once per epoch in `steps` steps.
        b1 = max(1, int(np.ceil(n1 / steps)))
        b2 = max(1, int(np.ceil(n2 / steps)))

        # Per-view sub-models sharing all layers/variables with self.model.
        model_v1 = Model(self.model.inputs[:2], self.model.outputs[0])
        model_v2 = Model(self.model.inputs[2:], self.model.outputs[1])

        mfloat = gpflow.default_float()  # model float (float32 under set_default_float)

        def make_ds(X, Y, W, b, seed):
            ds = tf.data.Dataset.from_tensor_slices(
                (X[:, 0].astype(np.int32), X[:, 1].astype(np.int32),
                 Y.astype(mfloat), W.astype(mfloat)))
            return ds.shuffle(min(len(Y), 100000), seed=seed,
                              reshuffle_each_iteration=True).repeat().batch(b)

        # `steps` batches/epoch; each view repeats so the smaller one cycles to fill them.
        ds = tf.data.Dataset.zip((make_ds(X_tr1, Y_tr1, sample_weight1, b1, 1),
                                  make_ds(X_tr2, Y_tr2, sample_weight2, b2, 2))
                                 ).take(steps).prefetch(tf.data.AUTOTUNE)

        opt = tf.keras.optimizers.Adam(self.lr)
        opt.build(self.model.trainable_variables)  # create slots outside the tf.function
        like1, like2 = self.loss_fn1, self.loss_fn2  # per-view (may differ, e.g. nb / bernoulli)
        train_vars = self.model.trainable_variables
        emb_vars = (self.emb1.trainable_variables + self.emb2.trainable_variables
                    + self.emb3.trainable_variables)
        f64 = tf.as_dtype(mfloat)  # accumulator/KL dtype = model float

        def _kl(gp_layer, kernel, num_data):
            kparams = list(kernel.trainable_parameters)
            log_prior = (tf.add_n([p.log_prior_density() for p in kparams])
                         if kparams else tf.constant(0.0, dtype=f64))
            return (gp_layer.prior_kl() - log_prior) / num_data

        # In-graph loss accumulator -> no per-step host sync (the killer for a Python loop).
        loss_acc = tf.Variable(0.0, dtype=f64, trainable=False)

        @tf.function  # compiled ONCE; the per-step graph stays small (no epoch unrolling).
        def step(b1d, b2d):
            i1, j1, y1, w1 = b1d
            i2, j2, y2, w2 = b2d
            with tf.GradientTape() as tape:
                f1 = model_v1((i1, j1), training=True)
                f2 = model_v2((i2, j2), training=True)
                # y -> (b, 1) to broadcast against the (b, num_latent_gps) GP output,
                # exactly as Keras fit/evaluate does internally.
                l1 = like1(tf.expand_dims(y1, -1), f1, sample_weight=w1)
                l2 = like2(tf.expand_dims(y2, -1), f2, sample_weight=w2)
                kl1 = _kl(self.gp_layer1, self.kernel1, self.num_data1)
                kl2 = _kl(self.gp_layer2, self.kernel2, self.num_data2)
                reg = self.emb_reg * tf.add_n(
                    [tf.reduce_sum(tf.square(v)) for v in emb_vars])
                loss = l1 + l2 + kl1 + kl2 + reg
            grads = tape.gradient(loss, train_vars)
            opt.apply_gradients(zip(grads, train_vars))
            loss_acc.assign_add(loss)

        history = {"loss": []}
        for epoch in range(n_iter):
            loss_acc.assign(0.0)
            for b1d, b2d in ds:           # C++-backed iteration over `steps` batches
                step(b1d, b2d)
            ep = float(loss_acc) / steps  # one host sync per epoch
            history["loss"].append(ep)
            print(f"Epoch {epoch + 1}/{n_iter}  (b1={b1}, b2={b2}, steps={steps})  "
                  f"loss={ep:.4f}")
        self.model.save_weights(os.path.join(self.save_path, "model_params.ckpt"))
        plt.plot(history["loss"])

        class _History:  # mimic keras History so downstream `hist.history["loss"]` works
            pass
        h = _History()
        h.history = history
        return h

    def train(self, X_tr1, Y_tr1, X_tr2=None, Y_tr2=None, labels=None, n_iter=None,
              sample_weight1=None, sample_weight2=None, efficient_multiview=False,
              extra_callbacks=()):
        """Train the model on the triple store(s).

        :param sample_weight1/2: optional (n,) per-triple weights for each view. Used by
            negative subsampling (``momogp.data.build_triple_store``) so the reweighted
            likelihood is an unbiased estimate of the full dense ELBO. Defaults to 1.
        :param efficient_multiview: if True (two-view only), train with a custom
            GradientTape loop (``_train_tape``) that batches the two views INDEPENDENTLY,
            the smaller view using a proportionally smaller batch — so it processes
            ~``n1 + n2`` rows/epoch instead of ~``2*max(n1, n2)`` and removes the padding
            loss-pollution. Objective is equivalent to Keras fit (verified to ~1e-6) and it
            honours ``self.lr`` (fit does not).
            **PERFORMANCE CAVEAT:** on CPU this is currently *slower* in wall-clock than the
            padded fit path — the hand-rolled per-step loop overhead (~17x/step here)
            outweighs the bounded ~2x FLOP saving. The triple-count reduction is real, so it
            may pay off on GPU (where per-step compute dominates dispatch), but benchmark
            before relying on it. For lopsided multi-view, prefer ``neg_sample_ratio`` (which
            shrinks BOTH views via the fast fit path) as the primary lever.
        """
        checkpoint_path = os.path.join(self.save_path, "model_params.ckpt")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path ,
                                                 save_weights_only=True,
                                                 verbose=1)
        # num_data must equal the size of the store fed to fit() (see __init__ note).
        self.num_data1 = int(len(Y_tr1))
        if self.K is not None and Y_tr2 is not None:
            self.num_data2 = int(len(Y_tr2))
        self.set_inducing_points(X_tr1, X_tr2)
        self.make_model()

        # Track whether the caller supplied weights, so the legacy path stays
        # bit-for-bit identical to before when they did not.
        weights_provided = (sample_weight1 is not None) or (sample_weight2 is not None)
        # Match targets/weights to the model's configured float (float64 default = no-op;
        # float32 under set_default_float halves data + activation memory on the GPU).
        mfloat = gpflow.default_float()
        Y_tr1 = np.asarray(Y_tr1, dtype=mfloat)
        sample_weight1 = (np.ones(len(Y_tr1), dtype=mfloat) if sample_weight1 is None
                          else np.asarray(sample_weight1, dtype=mfloat))
        if self.K is not None and Y_tr2 is not None:
            Y_tr2 = np.asarray(Y_tr2, dtype=mfloat)
            sample_weight2 = (np.ones(len(Y_tr2), dtype=mfloat) if sample_weight2 is None
                              else np.asarray(sample_weight2, dtype=mfloat))

        if self.K is not None:
            if efficient_multiview:
                # Independent per-view batching via a custom GradientTape loop: no padding,
                # and the smaller view uses a proportionally smaller batch, so total work
                # per epoch is ~n1+n2 instead of ~2*max(n1,n2). See _train_tape.
                return self._train_tape(X_tr1, Y_tr1, X_tr2, Y_tr2,
                                        sample_weight1, sample_weight2, n_iter)

            # Legacy padded path (kept for backward compatibility / reproducibility).
            raw_inputs = [X_tr1, X_tr2]
            padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs, padding="post")
            X_tr1_0padded = padded_inputs[0]
            X_tr2_0padded = padded_inputs[1]

            raw_inputs = [Y_tr1, Y_tr2]
            padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs, padding="post", dtype=np.float64)
            Y_tr1_0padded = padded_inputs[0]
            Y_tr2_0padded = padded_inputs[1]

            shuffle_ids1 = np.random.choice(range(X_tr1_0padded.shape[0]), X_tr1_0padded.shape[0], replace=False)
            X_tr_Shuffeled1 = np.copy(X_tr1_0padded[shuffle_ids1])
            Y_tr_Shuffeled1 = np.copy(Y_tr1_0padded[shuffle_ids1])
            shuffle_ids2 = np.random.choice(range(X_tr2_0padded.shape[0]), X_tr2_0padded.shape[0], replace=False)
            X_tr_Shuffeled2 = np.copy(X_tr2_0padded[shuffle_ids2])
            Y_tr_Shuffeled2 = np.copy(Y_tr2_0padded[shuffle_ids2])
            fit_kwargs = {}
            if weights_provided:
                # Pad weights with 0 so padded rows contribute nothing (fixes the
                # padding loss-pollution). Only applied when the caller passes weights,
                # so the no-weights legacy run is bit-for-bit unchanged.
                W1_padded = tf.keras.preprocessing.sequence.pad_sequences(
                    [sample_weight1, sample_weight2], padding="post", dtype=np.float64)[0]
                W2_padded = tf.keras.preprocessing.sequence.pad_sequences(
                    [sample_weight2, sample_weight1], padding="post", dtype=np.float64)[0]
                fit_kwargs["sample_weight"] = (np.copy(W1_padded[shuffle_ids1]),
                                               np.copy(W2_padded[shuffle_ids2]))
            hist = self.model.fit((X_tr_Shuffeled1[:, 0],X_tr_Shuffeled1[:, 1],X_tr_Shuffeled2[:, 0],X_tr_Shuffeled2[:, 1]), (Y_tr_Shuffeled1,Y_tr_Shuffeled2), epochs=n_iter, batch_size=self.batch_size, callbacks=[cp_callback, *extra_callbacks], **fit_kwargs)
            plt.plot(hist.history["loss"])
        else:
            shuffle_ids1 = np.random.choice(range(X_tr1.shape[0]), X_tr1.shape[0], replace=False)
            X_tr_Shuffeled1 = np.copy(X_tr1[shuffle_ids1])
            Y_tr_Shuffeled1 = np.copy(Y_tr1[shuffle_ids1])
            fit_kwargs = {}
            if weights_provided:
                fit_kwargs["sample_weight"] = np.copy(sample_weight1[shuffle_ids1])
            hist = self.model.fit([X_tr_Shuffeled1[:, 0],X_tr_Shuffeled1[:, 1]], Y_tr_Shuffeled1, epochs=n_iter, batch_size=self.batch_size, callbacks=[cp_callback, *extra_callbacks], **fit_kwargs)
            plt.plot(hist.history["loss"])
        return hist
        
        
    def load_model(self, X_tr1, X_tr2):
        self.set_inducing_points(X_tr1, X_tr2)
        self.make_model()    
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        checkpoint_path = os.path.join(self.save_path, "model_params.ckpt")  
        self.model.load_weights(checkpoint_path)
    