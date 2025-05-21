import torch
from collections import OrderedDict


def map_sb3_to_skrl(sb3_path, skrl_model):
    """
    Loads an SB3 .pth file (from a MultiInputPolicy with a custom extractor)
    and transfers matching weights into a SKRL model.

    Args:
        sb3_path: path to SB3 .pth file (e.g. "student_final/policy.pth")
        skrl_model: instance of GaussianNeuralNetworkConv or similar
    """
    print(f"[mapper] Loading SB3 weights from: {sb3_path}")
    sb3_state = torch.load(sb3_path, map_location="cpu")

    if "policy" in sb3_state:
        sb3_state = sb3_state["policy"]

    # Flatten all keys into a clean dict
    sb3_state = OrderedDict((k, v) for k, v in sb3_state.items())

    # Grab SKRL state dict
    skrl_state = skrl_model.state_dict()

    # Build a new dict with compatible keys
    matched = {}
    unmatched = []

    for skrl_k in skrl_state.keys():
        # Try to infer SB3 key based on common patterns
        if "mlp" in skrl_k:
            # e.g. mlp.0.weight <-> mlp.0.weight
            base_key = f"features_extractor.{skrl_k}"
        elif "encoder" in skrl_k:
            base_key = f"features_extractor.{skrl_k}"
        else:
            base_key = skrl_k

        if base_key in sb3_state and sb3_state[base_key].shape == skrl_state[skrl_k].shape:
            matched[skrl_k] = sb3_state[base_key]
        else:
            unmatched.append((skrl_k, base_key))

    # Load in matched
    print(f"[mapper] Transferring {len(matched)} weights into SKRL model")
    skrl_model.load_state_dict(matched, strict=False)

    if unmatched:
        print(f"[mapper] {len(unmatched)} keys unmatched:")
        for sk, sb in unmatched:
            print(f"  {sk}   <--/-->   {sb}")

    return skrl_model