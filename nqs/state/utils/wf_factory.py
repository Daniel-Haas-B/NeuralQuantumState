# from src.models import DS, Dummy, FFNN, RBM, VMC
import src.models as mods


def wf_factory(wf_type, **kwargs):
    wf_type = wf_type.lower() if isinstance(wf_type, str) else wf_type

    match wf_type:
        case "rbm":
            return mods.RBM(**kwargs)
        case "ffnn":
            return mods.FFNN(**kwargs)
        case "vmc":
            return mods.VMC(**kwargs)
        case "ds":
            return mods.DS(**kwargs)
        case "dummy":
            return mods.Dummy(**kwargs)
        case _:  # noqa
            raise NotImplementedError(
                f"No options for {wf_type}, Only the VMC, RBM, FFNN, DS (and Dummy) supported for now."
            )
