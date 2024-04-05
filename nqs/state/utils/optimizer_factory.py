from src.optimizers import Adagrad
from src.optimizers import Adam
from src.optimizers import Gd
from src.optimizers import RmsProp
from src.optimizers import Sr


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
