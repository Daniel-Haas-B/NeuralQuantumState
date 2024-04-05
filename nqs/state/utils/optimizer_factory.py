from nqs.optimizers import Adagrad
from nqs.optimizers import Adam
from nqs.optimizers import Gd
from nqs.optimizers import RmsProp
from nqs.optimizers import Sr


def optimizer_factory(opti_type, **kwargs):
    opti_type = opti_type.lower() if isinstance(opti_type, str) else opti_type

    match opti_type:
        case "gd":
            return Gd(**kwargs)
        case "adam":
            return Adam(**kwargs)
        case "rmsprop":
            return RmsProp(**kwargs)
        case "adagrad":
            return Adagrad(**kwargs)
        case "sr":
            return Sr(**kwargs)
        case _:  # noqa
            raise NotImplementedError(
                f"No options for {opti_type}, Only the gd, adam, rmsprop, adagrad and sr supported for now."
            )
