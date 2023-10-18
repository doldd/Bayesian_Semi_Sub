import torch
import torch.nn as nn

class cudaIter():
    def __init__(self, device, bs, data_set, collate_fn=torch.utils.data.dataloader.default_collate):
        loader = torch.utils.data.DataLoader(data_set, batch_size=len(data_set), shuffle=False, num_workers=0,
                                             persistent_workers=False, pin_memory=torch.cuda.is_available(),
                                             collate_fn=collate_fn)
        self.has_meta_ = False
        for data, y in loader:
            y = y.to(device=device)
            if type(data) is list or type(data) is tuple:
                img, meta = data
                img = img.to(device=device)
                meta = meta.to(device=device)
                self.dataset = torch.utils.data.TensorDataset(img, meta, y)
                self.has_meta_ = True
            else:
                img = data.to(device=device)
                self.dataset = torch.utils.data.TensorDataset(img, y)
        self.n_max = len(self.dataset) // bs
        assert (
                       len(self.dataset) % bs) == 0.0, f"equal sized batches required. Dataset shape: {len(self.dataset)}, bs: {bs}. bs should be one of the following: {[i for i in range(1, len(self.dataset), 1) if len(self.dataset) % i == 0.]}"
        self.bs = bs

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.n_max:
            self.n += 1
            if self.has_meta_:
                img, meta, y = self.dataset[(self.n - 1) * self.bs: self.n * self.bs]
                return (img, meta), y
            else:
                return self.dataset[(self.n - 1) * self.bs: self.n * self.bs]
        else:
            raise StopIteration

    def __len__(self):
        return self.n_max


class SemiAdditive:
    def __init__(self, output_dim, num_structure, ortho_layer_name_nn_head: str = None, *args, **kwargs):
        """

        :param num_structure: number of linear features
        :param ortho_layer_name_nn_head: define the name of the last layer (head layer of nn) which will be used for orthogonalisation. If it is None, no orthogonalisation is performed. For example ortho_layer_name_nn_head="lin[4]"
        :param args: Additional args for super class
        :param kwargs:
        """
        print("SemiAdditive")
        super().__init__(output_dim=output_dim, *args, **kwargs)
        self.num_structure_ = num_structure * output_dim
        self.structure_lin = nn.Linear(num_structure, output_dim, bias=False)
        self.save_hyperparameters("num_structure", "ortho_layer_name_nn_head", "output_dim")
        # Registration of the orthogonalisation layer, if the name of the last layer is given
        self.hook_handlerQ = None
        self.hook_handlerOrtho = None
        self.ortho_layer_nn_head_ = None
        if not ((ortho_layer_name_nn_head is None) or (ortho_layer_name_nn_head == "None")):
            self.ortho_layer_nn_head_ = eval("self." + ortho_layer_name_nn_head)
            self.batch_norm = nn.BatchNorm1d(self.ortho_layer_nn_head_.in_features, affine=False,
                                             track_running_stats=False)
            self.batch_norm_u = nn.BatchNorm1d(self.ortho_layer_nn_head_.in_features, affine=False,
                                               track_running_stats=False)
            self.batch_norm_X = nn.BatchNorm1d(num_structure, affine=False, track_running_stats=True)
            assert isinstance(self.ortho_layer_nn_head_,
                              nn.Module), f"Module {ortho_layer_name_nn_head} is not an instance of torch.nn.Module"
            self.hook_handlerOrtho = self.ortho_layer_nn_head_.register_forward_pre_hook(self._orthog_layer)
            self.hook_handlerQ = self.structure_lin.register_forward_pre_hook(self._compute_Q)
            # self.register_buffer("Q", torch.zeros(50,2))

    def remove_orthogonalisation_layer(self):
        if self.hook_handlerQ:
            self.hook_handlerQ.remove()
        if self.hook_handlerOrtho:
            print("Removed orthogonalisation layer")
            self.hook_handlerOrtho.remove()

    def add_orthogonalisation_layer(self):
        if self.ortho_layer_nn_head_:
            print("Added orthogonalisation layer")
            self.hook_handlerOrtho = self.ortho_layer_nn_head_.register_forward_pre_hook(self._orthog_layer)
            self.hook_handlerQ = self.structure_lin.register_forward_pre_hook(self._compute_Q)
        else:
            print("Can not perform orthogonalisation. No orthogonalisation layer was defined.")

    def _compute_Q(self, module, inputs):
        X = self.batch_norm_X(inputs[0])
        self.Q_mat, self.R_mat = torch.linalg.qr(X)
        # self.Q_mat, self.R_mat = torch.linalg.qr(inputs[0])
        return inputs

    def _orthog_layer(self, module, inputs):
        """
        Utilde = Uhat - QQTUhat
        """
        det_R = torch.diag(self.R_mat).prod()
        if det_R != 0.:
            Uhat = inputs[0]
            Uhat = self.batch_norm_u(Uhat)
            Projection_Matrix = self.Q_mat @ self.Q_mat.T
            Utilde = Uhat - Projection_Matrix @ Uhat
            # Utilde = self.batch_norm(Utilde)
            return Utilde, *inputs[1:]
        else:
            Utilde = self.batch_norm_u(inputs[0])
            print(
                "WARNING: No orthogonalisation performed. R matrix from QR decomposition not invertible (Not enough independent linear features)")
            return Utilde, *inputs[1:]

    def forward_nn(self, img):
        return super().forward(img)

    def forward_structure(self, x):
        return self.structure_lin(x)

    def forward(self, data):  # -> (bs x 1)
        u, x = data
        # u, x = (u.to(device=device), x.to(device=device)) if u.device != device else (u, x)
        return self.forward_structure(x) + self.forward_nn(
            u)  # Important! First compute structure forward pass such that the Q matrix is updated


class Subspace:
    def __init__(self, mean=torch.empty(0), cov_factor=torch.empty(0), *args, **kwargs):
        print("Subspace")
        super(Subspace, self).__init__(*args, **kwargs)
        self.rank = cov_factor.size(0)
        self.register_buffer('mean', mean.to(dtype=torch.float32))
        self.register_buffer('cov_factor', cov_factor.to(dtype=torch.float32))
        # self.nn_subspace_param = Parameter(torch.empty((self.rank,), dtype=mean.dtype, device=mean.device))
        # self.save_hyperparameters("mean", "cov_factor")

    def get_nn_param_vec(self, subspace_vector):
        return self.mean + self.cov_factor.t() @ subspace_vector

    def set_parameter_vector(self, subspace_vector: torch.Tensor):
        # p = self.nn_subspace_param
        # p.data.copy_(subspace_vector.view(p.shape))
        self.update_nn_params(self.get_nn_param_vec(subspace_vector))

    def update_nn_params(self, vec):
        p_idx = 0
        for p in self.dnn.parameters():
            # for n, p in self.named_parameters():
            #     if "nn_subspace_param" not in n and "structure_lin" not in n:
            # p.detach_()
            # p.data.fill_(vec[p_idx:p_idx + p.numel()].view(p.shape))
            p.data.copy_(vec[p_idx:p_idx + p.numel()].view(p.shape))
            # p = vec[p_idx:p_idx + p.numel()].view(p.shape)
            p_idx += p.numel()


class SemiAdditiveSubspace(SemiAdditive, Subspace):
    def __init__(self, num_structure, mean, cov_factor, opt_kwargs: dict = {}, *args, **kwargs):
        print("SemiAdditiveSubspace")
        # print(num_structure)
        # print(mean.shape)
        # print(cov_factor.shape)
        # kwargs.pop('num_structure', None)
        kwargs = dict(**opt_kwargs, **kwargs)
        self.nn_args = [args, kwargs]
        super(SemiAdditiveSubspace, self).__init__(mean=mean, cov_factor=cov_factor,
                                                   num_structure=num_structure,
                                                   *args,
                                                   **kwargs)
        # num_structure is only the nubmer of input features
        # and self.num_structer_ is the number of flattened structural parameters e.g. shape (5,3) = 15
        self.rank += self.num_structure_

    def set_parameter_vector(self, subspace_vector: torch.Tensor):
        super().set_parameter_vector(subspace_vector[:len(subspace_vector) - self.num_structure_])
        p = self.structure_lin.weight
        p.data.copy_(subspace_vector[-self.num_structure_:].view(p.shape))


def getModel(base_cls, num_structure, mean=None, cov_factor=None, *args, **kwargs):
    if num_structure >= 1:
        if mean is None or cov_factor is None:
            dynamic_class = type("SemiAdditiveModel", (SemiAdditive, base_cls), {})
            globals()["SemiAdditiveModel"] = dynamic_class
            return dynamic_class(num_structure=num_structure, *args, **kwargs)
        else:
            # def myreduce(self):
            #     print("myreduce")
            #     print(self.nn_args)
            #     print(self.__class__.__name__)
            #     globals()[self.__class__.__name__] = self.__class__
            #     # print(globals())
            #     return self.__class__, (
            #         self.num_structure_, self.mean, self.cov_factor, self.nn_args[1], *self.nn_args[0])

            dynamic_class = type("SemiAdditiveSubspaceModel", (SemiAdditiveSubspace, base_cls),
                                 {})
            # {"__reduce__": myreduce})
            globals()["SemiAdditiveSubspaceModel"] = dynamic_class
            return dynamic_class(num_structure=num_structure,
                                 mean=mean,
                                 cov_factor=cov_factor,
                                 *args, **kwargs)
    else:
        if mean is None or cov_factor is None:
            return base_cls(*args, **kwargs)
        else:
            dynamic_class = type("SubspaceModel", (Subspace, base_cls), {})
            globals()["SubspaceModel"] = dynamic_class
            return dynamic_class(mean=mean,
                                 cov_factor=cov_factor,
                                 *args, **kwargs)


# class pyroLinRegSubspace(PyroModule):
#     def __init__(self, fc_model: Subspace):
#         super(pyroLinRegSubspace, self).__init__()
#         self.base_model_ = fc_model
#         # del self.base_model_.test_metrics
#         # del self.base_model_.valid_metrics
#         # del self.base_model_.loss_fn
#         # self._data_std_prior = dist.Gamma(torch.tensor(2., device=fc_model.device),
#         #                                   torch.tensor(2., device=fc_model.device))  # prior for scale parameter
#         # self.scale = PyroSample(prior=self._data_std_prior)
#         self.subspace_param = PyroSample(prior=dist.Normal(torch.tensor(0., device=fc_model.device),
#                                                            torch.tensor(5., device=fc_model.device)).expand(
#             (fc_model.rank,)).to_event(1))

#     def forward(self, img, meta, yt):
#         self.base_model_.set_parameter_vector(self.subspace_param)
#         loc = self.base_model_(img, meta).squeeze()
#         # scale_ = pyro.sample("scale", self._data_std_prior)
#         with pyro.plate("data", img.shape[0]):
#             distri = dist.Normal(loc, .1)
#             likely = pyro.sample("obs", distri, obs=yt)
#         return likely

#     def __reduce__(self):
#         return self.__class__, (self.base_model_,)




if __name__ == '__main__':
    print("test")
