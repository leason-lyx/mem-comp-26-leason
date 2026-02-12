"""
The script is used to evaluate the performance of the SWEAP Pro agent with local Docker.

This evaluation script:
1. Takes a CSV file containing test cases and a JSON file containing patches
2. Runs each patch in a local Docker environment using Docker Hub images
3. Executes the tests using local run scripts and collects results
4. Calculates overall accuracy based on test pass/fail status

Usage:
python swe_bench_pro_eval.py \
    --raw_sample_path=data.csv \
    --patch_path={OUTPUT}/gold_patches.json \
    --output_dir={OUTPUT}/ \
    --scripts_dir=run_scripts \
    --num_workers=100 \
    --dockerhub_username=your-username

It expects:
- Local run scripts in run_scripts/{instance_id}/run_script.sh
- Local parser scripts in run_scripts/{instance_id}/parser.py
- CSV file with columns: instance_id, before_repo_set_cmd, selected_test_files_to_run, 
  base_commit, base_dockerfile, instance_dockerfile, FAIL_TO_PASS, PASS_TO_PASS

And the generated patch file (gold_patches.json) should have the following format:
[
    {
        "instance_id": "unique_id",
        "patch": "git patch content",
        "prefix": "optional_prefix"
    },
    ...
]
"""

import argparse
import concurrent.futures
import json
import os
import platform as py_platform
import re

try:
    import docker  # Optional: used for local Docker execution
except Exception:
    docker = None
import pandas as pd
from tqdm import tqdm


def get_dockerhub_image_uri(uid, dockerhub_username, repo_name=""):
    """Fallback helper for harness-local usage when helper_code is unavailable."""
    repo_base, repo_name_only = repo_name.lower().split("/")
    hsh = uid.replace("instance_", "")

    if (
        uid
        == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan"
    ):
        repo_name_only = "element-web"  # Keep full name for this one case
    elif "element-hq" in repo_name.lower() and "element-web" in repo_name.lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]

    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]

    return f"{dockerhub_username}/sweap-images:{tag}"


# Credit: prabhuteja12
def load_base_docker(iid):
    with open(f"dockerfiles/base_dockerfile/{iid}/Dockerfile") as fp:
        return fp.read()


def instance_docker(iid):
    with open(f"dockerfiles/instance_dockerfile/{iid}/Dockerfile") as fp:
        return fp.read()


def load_local_script(scripts_dir, instance_id, script_name):
    """Load a script file from local scripts directory."""
    script_path = os.path.join(scripts_dir, instance_id, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    with open(script_path, "r") as f:
        return f.read()


def strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch."""
    if not patch:
        return patch

    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)

    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)

    return "".join(kept)


def create_entryscript(sample):
    before_repo_set_cmd = sample["before_repo_set_cmd"].strip().split("\n")[-1]
    selected_test_files_to_run = ",".join(eval(sample["selected_test_files_to_run"]))
    base_commit = sample["base_commit"]
    base_dockerfile = load_base_docker(sample["instance_id"])
    instance_dockerfile = instance_docker(sample["instance_id"])

    # Extract ENV commands from dockerfiles
    env_cmds = []
    for dockerfile_content in [base_dockerfile, instance_dockerfile]:
        for line in dockerfile_content.split("\n"):
            line = line.strip()
            if line.startswith("ENV"):
                # Convert ENV commands to export statements
                env_cmd = line.replace("ENV", "export", 1)
                env_cmds.append(env_cmd)

    env_cmds = "\n".join(env_cmds)

    entry_script = f"""
{env_cmds}
# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/patch.diff
{before_repo_set_cmd}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_test_files_to_run} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""
    return entry_script


def create_dockerhub_tag(uid, repo_name=""):
    """
    Convert instance_id and repo name to Docker Hub compatible tag format.
    This must match the format used in the upload script.

    Args:
        uid (str): The instance_id (e.g., "django__django-12345")
        repo_name (str): The repository name from ECR (e.g., "sweap-images/nodebb.nodebb")

    Returns:
        str: Docker Hub compatible tag (e.g., "nodebb-nodebb-12345")
    """
    if repo_name:
        # For "NodeBB/NodeBB" -> repo_base="nodebb", repo_name="nodebb"
        # Format: {repo_base}.{repo_name}-{OriginalCase}__{OriginalCase}-{hash}-{version}
        # Example: nodebb.nodebb-NodeBB__NodeBB-7b8bffd763e2155cf88f3ebc258fa68ebe18188d-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e
        repo_base, repo_name_only = repo_name.lower().split("/")
        # Keep original case for the instance_id part (after removing "instance_" prefix)
        hsh = uid.replace("instance_", "")
        return f"{repo_base}.{repo_name_only}-{hsh}"
    else:
        image_name = "default"

    # Extract the tag part from the instance ID
    # For UIDs that start with a pattern like "django__django-", extract everything after position 9
    if "__" in uid and len(uid) > 9:
        tag_part = uid[9:]  # Skip the first 9 characters (e.g., "django__")
    else:
        tag_part = uid

    return f"{image_name}-{tag_part}"


def prepare_run(uid, output_dir, prefix, redo):
    uid_dir = os.path.join(output_dir, uid)
    os.makedirs(uid_dir, exist_ok=True)
    output_path = os.path.join(uid_dir, f"{prefix}_output.json")
    if not redo and os.path.exists(output_path):
        print(f"Skipping {uid} - output already exists")
        with open(output_path, "r") as f:
            return json.load(f), output_path, os.path.join(uid_dir, "workspace")
    workspace_dir = os.path.join(uid_dir, "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    return None, output_path, workspace_dir


def write_patch_snapshot(output_dir, uid, prefix, patch):
    with open(os.path.join(output_dir, uid, f"{prefix}_patch.diff"), "w") as f:
        f.write(patch)


def assemble_workspace_files(uid, scripts_dir, patch, sample):
    run_script = load_local_script(scripts_dir, uid, "run_script.sh")
    parser_script = load_local_script(scripts_dir, uid, "parser.py")
    entryscript_content = create_entryscript(sample)
    cleaned_patch = strip_binary_hunks(patch)
    if cleaned_patch != patch:
        print(f"Stripped binary diff hunks from patch for {uid}")

    files = {
        "patch.diff": cleaned_patch,
        "run_script.sh": run_script,
        "parser.py": parser_script,
        "entryscript.sh": entryscript_content,
    }
    return files, entryscript_content


def write_files_local(workspace_dir, files):
    for rel_path, content in files.items():
        dst = os.path.join(workspace_dir, rel_path)
        with open(dst, "w") as f:
            f.write(content)


def save_entryscript_copy(output_dir, uid, prefix, entryscript_content):
    with open(os.path.join(output_dir, uid, f"{prefix}_entryscript.sh"), "w") as f:
        f.write(entryscript_content if entryscript_content is not None else "")


def collect_outputs_local(workspace_dir, output_dir, uid, prefix):
    def _copy_safe(src_name, dest_name):
        src_path = os.path.join(workspace_dir, src_name)
        dest_path = os.path.join(output_dir, uid, dest_name)
        try:
            with open(src_path, "r") as f_in:
                content = f_in.read()
        except FileNotFoundError:
            content = ""
        with open(dest_path, "w") as f_out:
            f_out.write(content if content is not None else "")

    _copy_safe("stdout.log", f"{prefix}_stdout.log")
    _copy_safe("stderr.log", f"{prefix}_stderr.log")

    # Then try to read output.json
    try:
        with open(os.path.join(workspace_dir, "output.json"), "r") as f_in:
            output = json.load(f_in)
            with open(os.path.join(output_dir, uid, f"{prefix}_output.json"), "w") as f:
                json.dump(output, f)
            return output
    except FileNotFoundError:
        print(
            f"Warning: output.json not found for {uid}. Check {prefix}_stdout.log and {prefix}_stderr.log for details"
        )
        return None


def eval_with_docker(
    patch,
    sample,
    output_dir,
    dockerhub_username,
    scripts_dir,
    prefix="",
    redo=False,
    block_network=False,
    docker_platform=None,
):
    if docker is None:
        raise RuntimeError(
            "docker SDK is not installed. Install via 'pip install docker'"
        )
    uid = sample["instance_id"]
    existing_output, output_path, workspace_dir = prepare_run(
        uid, output_dir, prefix, redo
    )
    if existing_output is not None:
        return existing_output

    print(f"Running local-docker evaluation for {uid}")

    try:
        try:
            files, entryscript_content = assemble_workspace_files(
                uid, scripts_dir, patch, sample
            )
        except FileNotFoundError as e:
            print(f"Error loading scripts for {uid}: {e}")
            return None
        write_files_local(workspace_dir, files)
        write_patch_snapshot(output_dir, uid, prefix, patch)

        # Run container via Docker SDK
        dockerhub_image_uri = get_dockerhub_image_uri(
            uid, dockerhub_username, sample.get("repo", "")
        )
        # print(f"Using Docker Hub image: {dockerhub_image_uri}")

        client = docker.from_env()
        try:
            if docker_platform:
                client.images.pull(dockerhub_image_uri, platform=docker_platform)
            else:
                client.images.pull(dockerhub_image_uri)
        except Exception as pull_err:
            # If pull fails, fall back to a local image if present; otherwise, fail this run
            try:
                client.images.get(dockerhub_image_uri)
                print(f"Using locally available image: {dockerhub_image_uri}")
            except Exception:
                print(f"Failed to pull or find image locally for {uid}: {pull_err}")
                return None

        abs_workspace_dir = os.path.abspath(workspace_dir)
        volumes = {abs_workspace_dir: {"bind": "/workspace", "mode": "rw"}}
        run_kwargs = {
            "volumes": volumes,
            "detach": True,
            "remove": True,
            "entrypoint": "/bin/bash",  # Override image entrypoint
            "command": ["-c", "bash /workspace/entryscript.sh"],
        }
        if block_network:
            run_kwargs["network_mode"] = "none"
        # Optional platform override (useful on Apple Silicon)
        if docker_platform:
            run_kwargs["platform"] = docker_platform

        container = client.containers.run(
            dockerhub_image_uri,
            **run_kwargs,
        )

        result = container.wait()
        status_code = result.get("StatusCode", 1) if isinstance(result, dict) else 1
        if status_code != 0:
            print(f"Entryscript failed for {uid} with return code: {status_code}")
        # Collect outputs and logs, and save entryscript for reference
        output = collect_outputs_local(workspace_dir, output_dir, uid, prefix)
        if output is None:
            return None
        save_entryscript_copy(output_dir, uid, prefix, entryscript_content)

        return output
    except Exception as e:
        print(f"Error in eval_with_docker for {uid}: {repr(e)}")
        print(f"Error type: {type(e)}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SWEAP Pro evaluations using local Docker with Docker Hub images and local scripts"
    )
    parser.add_argument(
        "--raw_sample_path", required=True, help="Path to the raw sample CSV/JSONL file"
    )

    patch_input_group = parser.add_mutually_exclusive_group(required=True)
    patch_input_group.add_argument(
        "--patch_path",
        help="Path to one patch JSON file",
    )
    patch_input_group.add_argument(
        "--patch_dir",
        help="Directory containing multiple patch JSON files; will write merged eval_results.json into this folder",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to store per-instance evaluation artifacts. "
        "Single-file mode default: parent dir of --patch_path. "
        "Batch mode default: --patch_dir",
    )
    parser.add_argument(
        "--dockerhub_username",
        required=True,
        help="Docker Hub username where sweap-images repository is located",
    )
    parser.add_argument(
        "--scripts_dir",
        required=True,
        help="Directory containing local run scripts (e.g., scripts/run_scripts)",
    )
    parser.add_argument(
        "--docker_platform",
        default=None,
        help="Docker platform override, e.g., linux/amd64; defaults to auto-detect",
    )
    parser.add_argument(
        "--redo", action="store_true", help="Redo evaluations even if output exists"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=50,
        help="Number of workers to run evaluations in parallel",
    )
    parser.add_argument(
        "--block_network",
        action="store_true",
        help="Block network access inside container",
    )
    return parser.parse_args()


def compute_accuracy(eval_results):
    if not eval_results:
        return 0.0
    return sum(eval_results.values()) / len(eval_results)


def load_raw_samples(raw_sample_path):
    if raw_sample_path.endswith(".jsonl"):
        raw_sample_df = pd.read_json(raw_sample_path, lines=True)
    else:
        raw_sample_df = pd.read_csv(raw_sample_path)

    raw_sample_df = raw_sample_df.fillna("")
    return raw_sample_df.set_index("instance_id", drop=False)


def load_patches_file(patch_path):
    with open(patch_path, "r") as f:
        patches_to_run = json.load(f)
    if not isinstance(patches_to_run, list):
        raise ValueError(f"Patch file must be a JSON list: {patch_path}")
    return patches_to_run


def list_patch_files(patch_dir):
    def _looks_like_patch_file(path):
        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except Exception:
            return False
        if not isinstance(payload, list):
            return False
        return any(isinstance(item, dict) and "instance_id" in item for item in payload)

    patch_files = []
    skip_prefixes = ("projects", "candidates")
    for name in sorted(os.listdir(patch_dir)):
        path = os.path.join(patch_dir, name)
        if not os.path.isfile(path):
            continue
        if not name.endswith(".json"):
            continue
        if name == "eval_results.json":
            continue
        stem = os.path.splitext(name)[0].lower()
        if stem.startswith(skip_prefixes):
            continue
        if not _looks_like_patch_file(path):
            continue
        patch_files.append(path)
    return patch_files


def evaluate_patch_list(
    patches_to_run,
    raw_sample_df,
    output_dir,
    args,
    eval_fn,
    docker_platform=None,
    run_label="",
):
    eval_results = {}

    valid_patches = []
    missing_instances = []
    malformed_samples = 0

    for patch_sample in patches_to_run:
        if not isinstance(patch_sample, dict) or "instance_id" not in patch_sample:
            malformed_samples += 1
            continue
        instance_id = patch_sample["instance_id"]
        if instance_id in raw_sample_df.index:
            valid_patches.append(patch_sample)
        else:
            missing_instances.append(instance_id)

    if malformed_samples:
        print(f"Warning: skipped {malformed_samples} malformed patch samples")

    if missing_instances:
        print(
            f"Warning: Found {len(missing_instances)} patch instances not in raw sample data:"
        )
        for missing_id in missing_instances[:5]:
            print(f"  - {missing_id}")
        if len(missing_instances) > 5:
            print(f"  ... and {len(missing_instances) - 5} more")
        print(
            f"Proceeding with {len(valid_patches)} valid patches out of {len(patches_to_run)} total patches"
        )

    if not valid_patches:
        return {
            "eval_results": eval_results,
            "accuracy": 0.0,
            "total_patches": len(patches_to_run),
            "valid_patches": 0,
            "missing_instances": missing_instances,
        }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        future_to_patch = {
            executor.submit(
                eval_fn,
                patch_sample.get("model_patch", patch_sample.get("patch", "")),
                raw_sample_df.loc[patch_sample["instance_id"]],
                output_dir,
                args.dockerhub_username,
                args.scripts_dir,
                prefix=patch_sample.get("prefix", ""),
                redo=args.redo,
                block_network=args.block_network,
                docker_platform=docker_platform,
            ): patch_sample
            for patch_sample in valid_patches
        }

        pbar = tqdm(
            concurrent.futures.as_completed(future_to_patch), total=len(valid_patches)
        )
        for future in pbar:
            patch_sample = future_to_patch[future]
            try:
                output = future.result()
                if output is None:
                    print(f'Evaluation for {patch_sample["instance_id"]} returned None')
                    eval_results[patch_sample["instance_id"]] = False
                else:
                    instance_id = patch_sample["instance_id"]
                    if instance_id not in raw_sample_df.index:
                        print(
                            f"Warning: Instance {instance_id} not found in raw sample data, skipping"
                        )
                        eval_results[instance_id] = False
                    else:
                        raw_sample = raw_sample_df.loc[instance_id]
                        passed_tests = {
                            x["name"]
                            for x in output["tests"]
                            if x["status"] == "PASSED"
                        }
                        f2p = set(eval(raw_sample["fail_to_pass"]))
                        p2p = set(eval(raw_sample["pass_to_pass"]))
                        eval_results[instance_id] = (f2p | p2p) <= passed_tests

                current_accuracy = compute_accuracy(eval_results)
                if run_label:
                    pbar.set_description(
                        f"[{run_label}] Accuracy: {current_accuracy:.2%}"
                    )
                else:
                    pbar.set_description(f"Accuracy: {current_accuracy:.2%}")
            except Exception as exc:
                print(
                    f'Evaluation for {patch_sample["instance_id"]} generated an exception: {exc}'
                )
                eval_results[patch_sample["instance_id"]] = False
                current_accuracy = compute_accuracy(eval_results)
                if run_label:
                    pbar.set_description(
                        f"[{run_label}] Accuracy: {current_accuracy:.2%}"
                    )
                else:
                    pbar.set_description(f"Accuracy: {current_accuracy:.2%}")

    return {
        "eval_results": eval_results,
        "accuracy": compute_accuracy(eval_results),
        "total_patches": len(patches_to_run),
        "valid_patches": len(valid_patches),
        "missing_instances": missing_instances,
    }


def detect_docker_platform(args):
    if args.docker_platform is not None:
        return args.docker_platform
    try:
        if py_platform.machine().lower() in {"arm64", "aarch64"}:
            return "linux/amd64"
    except Exception:
        return None
    return None


def main():
    args = parse_args()

    raw_sample_df = load_raw_samples(args.raw_sample_path)
    eval_fn = eval_with_docker
    docker_platform = detect_docker_platform(args)

    if args.patch_dir:
        patch_dir = os.path.abspath(args.patch_dir)
        patch_files = list_patch_files(patch_dir)
        if not patch_files:
            raise FileNotFoundError(f"No JSON patch files found in: {patch_dir}")

        artifacts_root = (
            os.path.abspath(args.output_dir) if args.output_dir else patch_dir
        )
        os.makedirs(artifacts_root, exist_ok=True)

        merged_results = {}
        for patch_path in patch_files:
            agent_name = os.path.splitext(os.path.basename(patch_path))[0]
            run_output_dir = os.path.join(artifacts_root, agent_name)
            os.makedirs(run_output_dir, exist_ok=True)
            print(f"\n===== Evaluating {agent_name} =====")

            patches_to_run = load_patches_file(patch_path)
            run_result = evaluate_patch_list(
                patches_to_run=patches_to_run,
                raw_sample_df=raw_sample_df,
                output_dir=run_output_dir,
                args=args,
                eval_fn=eval_fn,
                docker_platform=docker_platform,
                run_label=agent_name,
            )

            passed = sum(run_result["eval_results"].values())
            total_evald = len(run_result["eval_results"])
            print(
                f"[{agent_name}] accuracy={run_result['accuracy']:.2%} "
                f"({passed}/{total_evald})"
            )

            merged_results[agent_name] = {
                "patch_path": patch_path,
                "output_dir": run_output_dir,
                "accuracy": run_result["accuracy"],
                "total_patches": run_result["total_patches"],
                "valid_patches": run_result["valid_patches"],
                "evaluated_instances": total_evald,
                "passed_instances": passed,
                "missing_instances": run_result["missing_instances"],
                "instance_results": run_result["eval_results"],
            }

        merged_output_path = os.path.join(patch_dir, "eval_results.json")
        with open(merged_output_path, "w") as f:
            json.dump(merged_results, f, indent=2)
        print(f"\nSaved merged eval results: {merged_output_path}")
        return

    patch_path = os.path.abspath(args.patch_path)
    default_output_dir = os.path.dirname(patch_path) or "."
    output_dir = (
        os.path.abspath(args.output_dir) if args.output_dir else default_output_dir
    )
    os.makedirs(output_dir, exist_ok=True)

    patches_to_run = load_patches_file(patch_path)
    run_result = evaluate_patch_list(
        patches_to_run=patches_to_run,
        raw_sample_df=raw_sample_df,
        output_dir=output_dir,
        args=args,
        eval_fn=eval_fn,
        docker_platform=docker_platform,
    )

    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(run_result["eval_results"], f)
    print("Overall accuracy: ", run_result["accuracy"])


if __name__ == "__main__":
    main()
