from nqs.models import Dummy
from nqs.models import FFNN
from nqs.models import RBM
from nqs.models import VMC


def wf_factory(wf_type, **kwargs):
    wf_type = wf_type.lower() if isinstance(wf_type, str) else wf_type

    match wf_type:
        case "rbm":
            return RBM(**kwargs)
        case "ffnn":
            return FFNN(**kwargs)
        case "vmc":
            return VMC(**kwargs)
        case "dummy":
            return Dummy(**kwargs)
        case _:  # noqa
            raise NotImplementedError(
                "Only the VMC, RBM and FFNN (and Dummy) supported for now."
            )
