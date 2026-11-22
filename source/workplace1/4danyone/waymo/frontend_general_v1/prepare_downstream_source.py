"""Stage references to complete source videos without copying large assets."""
import argparse
import hashlib
import json
from pathlib import Path

G = Path(__file__).resolve().parent
V = G / "validation12"
D = V / "downstream"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample")
    parser.add_argument("arm", choices=["G0", "G2"])
    args = parser.parse_args()
    cfg = read(D / "PREP_PROTOCOL.json")
    assert args.sample in cfg["subjects"]
    parent = V / "generation" / args.sample / "G2"
    validation = parent / "validation.json"
    assert read(validation)["state"] == "complete_decoding_and_stage_evidence_verified"
    old = Path(cfg["sources"][args.sample])
    assert read(V / "generation/old_checks" / f"{args.sample}.json")["sample"] == args.sample
    source = old if args.arm == "G0" else parent / "outputs/turbo"
    metadata = source / ("generation.json" if args.arm == "G0" else "metadata.json")
    generation = read(metadata)
    assert generation["generation"]["seed"] == 42
    assert generation["output"]["target_views"] == 24
    assert generation["output"]["frames_per_video"] == cfg["frames"][args.sample]
    cameras = read(source / "cameras.json")
    assert cameras["world_frame"]["name"] == "root_local_human_world"
    assert [c["camera_id"] for c in cameras["cameras"]] == list(range(24))
    root_motion = old / "root_motion.npz" if args.arm == "G0" else parent / "preflight/root_motion.npz"
    destination = D / "sources" / f"{args.sample}_{args.arm}"
    destination.mkdir(exist_ok=False)
    for name, path in (
        ("videos", source / "videos"), ("skeletons", source / "skeletons"),
        ("metadata.json", metadata), ("cameras.json", source / "cameras.json"),
        ("root_motion.npz", root_motion),
    ):
        assert path.exists()
        (destination / name).symlink_to(path, target_is_directory=path.is_dir())
    hashes = {str(p): sha(p) for p in [metadata, source / "cameras.json", root_motion, validation]}
    hashes.update({str(p): sha(p) for p in sorted((source / "videos").glob("*.mp4"))})
    assert len(list((source / "videos").glob("*.mp4"))) == 24
    receipt = dict(
        state="references_staged_not_reconstructed", sample=args.sample, arm=args.arm,
        input_sha256=hashes, seed=42, quality_approved=False,
        source_scale="Estimated SMPL convention; not validated metric Waymo world motion.",
        comparison="Whole old/new frontend; each arm keeps its own cameras and temporal geometry.",
        root_sidecar=str(root_motion),
    )
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"sample": args.sample, "arm": args.arm, "state": receipt["state"]}))


if __name__ == "__main__":
    main()
