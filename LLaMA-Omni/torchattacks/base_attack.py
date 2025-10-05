import time
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, TensorDataset

def wrapper_method(func):
    def wrapper_func(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        for atk in self.__dict__.get("_attacks").values():
            eval("atk." + func.__name__ + "(*args, **kwargs)")
        return result
    return wrapper_func

class Attack(object):
    """
    Base class for all attacks.

    Note:
        Automatically sets device to the device where the given model is.
        Changes training mode to eval during attack process.
        To change this, see `set_model_training_mode`.
    """
    def __init__(self, name, model):
        """
        Initializes internal attack state.

        Args:
            name (str): Name of attack.
            model (torch.nn.Module): Model to attack.
        """
        self.attack = name
        self._attacks = OrderedDict()
        self.set_model(model)
        try:
            self.device = next(model.parameters()).device
        except Exception:
            self.device = None
            print("Failed to set device automatically, please try set_device() manual.")
        self.attack_mode = "default"
        self.supported_mode = ["default"]
        self.targeted = False
        self._target_map_function = None
        self.normalization_used = None
        self._normalization_applied = None
        if self.model.__class__.__name__ == "RobModel":
            self._set_rmodel_normalization_used(model)
        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False

    def forward(self, inputs, labels=None, *args, **kwargs):
        """
        Defines the computation performed at every call. Should be overridden by all subclasses.
        """
        raise NotImplementedError

    @wrapper_method
    def set_model(self, model):
        self.model = model
        self.model_name = model.__class__.__name__

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if isinstance(inputs, list):
            if self._normalization_applied is False:
                inputs[0] = self.normalize(inputs[0])
            logits = self.model(inputs)
        else:
            if self._normalization_applied is False:
                inputs = self.normalize(inputs)
            logits = self.model(inputs)
        return logits

    @wrapper_method
    def _set_normalization_applied(self, flag):
        self._normalization_applied = flag

    @wrapper_method
    def set_device(self, device):
        self.device = device

    @wrapper_method
    def _set_rmodel_normalization_used(self, model):
        mean = getattr(model, "mean", None)
        std = getattr(model, "std", None)
        if (mean is not None) and (std is not None):
            if isinstance(mean, torch.Tensor):
                mean = mean.cpu().numpy()
            if isinstance(std, torch.Tensor):
                std = std.cpu().numpy()
            if (mean != 0).all() or (std != 1).all():
                self.set_normalization_used(mean, std)

    @wrapper_method
    def set_normalization_used(self, mean, std):
        self.normalization_used = {}
        n_channels = len(mean)
        mean = torch.tensor(mean).reshape(1, n_channels, 1, 1)
        std = torch.tensor(std).reshape(1, n_channels, 1, 1)
        self.normalization_used["mean"] = mean
        self.normalization_used["std"] = std
        self._set_normalization_applied(True)

    def normalize(self, inputs):
        if isinstance(inputs, list):
            mean = self.normalization_used["mean"].to(inputs[0].device)
            std = self.normalization_used["std"].to(inputs[0].device)
            outputs = []
            for input_ in inputs:
                output = (input_ - mean)/std
                outputs.append(output)
            return outputs
        else:
            mean = self.normalization_used["mean"].to(inputs.device)
            std = self.normalization_used["std"].to(inputs.device)
            return (inputs - mean) / std

    def inverse_normalize(self, inputs):
        if isinstance(inputs, list):
            mean = self.normalization_used["mean"].to(inputs[0].device)
            std = self.normalization_used["std"].to(inputs[0].device)
            outputs = []
            for input_ in inputs:
                output = input_ * std + mean
                outputs.append(output)
            return outputs
        else:
            mean = self.normalization_used["mean"].to(inputs.device)
            std = self.normalization_used["std"].to(inputs.device)
            return inputs * std + mean

    def get_mode(self):
        """Get attack mode."""
        return self.attack_mode

    @wrapper_method
    def set_mode_default(self):
        self.attack_mode = "default"
        self.targeted = False
        print("Attack mode is changed to 'default.'")

    @wrapper_method
    def _set_mode_targeted(self, mode, quiet):
        if "targeted" not in self.supported_mode:
            raise ValueError("Targeted mode is not supported.")
        self.targeted = True
        self.attack_mode = mode
        if not quiet:
            print(f"Attack mode is changed to '{mode}'.")

    @wrapper_method
    def set_mode_targeted_by_function(self, target_map_function, quiet=False):
        self._set_mode_targeted("targeted(custom)", quiet)
        self._target_map_function = target_map_function

    @wrapper_method
    def set_mode_targeted_random(self, quiet=False):
        self._set_mode_targeted("targeted(random)", quiet)
        self._target_map_function = self.get_random_target_label

    @wrapper_method
    def set_mode_targeted_least_likely(self, kth_min=1, quiet=False):
        self._set_mode_targeted("targeted(least-likely)", quiet)
        assert kth_min > 0
        self._kth_min = kth_min
        self._target_map_function = self.get_least_likely_label

    @wrapper_method
    def set_mode_targeted_by_label(self, quiet=False):
        self._set_mode_targeted("targeted(label)", quiet)
        self._target_map_function = "function is a string"

    @wrapper_method
    def set_model_training_mode(self, model_training=False, batchnorm_training=False, dropout_training=False):
        self._model_training = model_training
        self._batchnorm_training = batchnorm_training
        self._dropout_training = dropout_training

    @wrapper_method
    def _change_model_mode(self, given_training):
        if self._model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not self._batchnorm_training:
                    if "BatchNorm" in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if "Dropout" in m.__class__.__name__:
                        m = m.eval()
        else:
            self.model.eval()

    @wrapper_method
    def _recover_model_mode(self, given_training):
        if given_training:
            self.model.train()

    def save(self, data_loader, save_path=None, verbose=True, return_verbose=False, save_predictions=False, save_clean_inputs=False, save_type="float"):
        # Implementation omitted for brevity. Add as needed.
        pass

    def __call__(self, inputs, labels=None, *args, **kwargs):
        given_training = self.model.training
        self._change_model_mode(given_training)
        if self._normalization_applied is True:
            inputs = self.inverse_normalize(inputs)
            self._set_normalization_applied(False)
            adv_inputs = self.forward(inputs, labels, *args, **kwargs)
            adv_inputs = self.normalize(adv_inputs)
            self._set_normalization_applied(True)
        else:
            adv_inputs = self.forward(inputs, labels, *args, **kwargs)
        self._recover_model_mode(given_training)
        return adv_inputs

    def __repr__(self):
        info = self.__dict__.copy()
        del_keys = ["model", "attack", "supported_mode"]
        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)
        for key in del_keys:
            del info[key]
        info["attack_mode"] = self.attack_mode
        info["normalization_used"] = (
            True if self.normalization_used is not None else False
        )
        return (
            self.attack
            + "("
            + ", ".join(f"{key}={val}" for key, val in info.items())
            + ")"
        )

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        attacks = self.__dict__.get("_attacks")
        def get_all_values(items, stack=[]):
            if items not in stack:
                stack.append(items)
                if isinstance(items, list) or isinstance(items, dict):
                    if isinstance(items, dict):
                        items = list(items.keys()) + list(items.values())
                    for item in items:
                        yield from get_all_values(item, stack)
                else:
                    if isinstance(items, Attack):
                        yield items
            else:
                if isinstance(items, Attack):
                    yield items
        for num, value in enumerate(get_all_values(value)):
            attacks[name + "." + str(num)] = value
            for subname, subvalue in value.__dict__.get("_attacks").items():
                attacks[name + "." + subname] = subvalue 