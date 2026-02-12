from dataclasses import dataclass, asdict
import pandas as pd
from pathlib import Path
import shutil
import psutil
import random
import json
from contextlib import contextmanager
import os
import docker
import threading
import multiprocessing
import secrets
import requests
import datetime
import traceback
import time
from pwd import getpwnam
from dotenv import load_dotenv
from typing import Callable

load_dotenv()

COLLECT_ONLY = os.getenv("COLLECT_ONLY", "0") == "1"

if os.geteuid() != 0 and not COLLECT_ONLY:
    raise PermissionError("please run as root!")
    # because file operations may fail for folders mounted to docker

VERBOSE = int(os.getenv("VERBOSE", "2"))
FILE_OWNER_UID = getpwnam(os.getenv("FILE_OWNER", "root")).pw_uid
RUN_NAME = os.getenv("RUN_NAME", "default").strip() or "default"
RESULT_PATH = Path("results") / RUN_NAME
WORK_PATH = Path("workdir") / RUN_NAME
SSHBOX_SCRIPT_PATH = Path("../sshbox_scripts")
LITELLM_TRAJ_PATH = Path(os.getenv("LITELLM_TRAJ_DIR", "../play/trajs"))
LITELLM_BASEURL_IN_DOCKER = "http://litellm_app:4000"
LITELLM_CONTROLLER_BASEURL = "http://127.0.0.1:4001"
ENABLE_LINKOS_MIRROR = os.getenv("ENABLE_LINKOS_MIRROR", "1") == "1"
LINKOS_MIRROR_MAP = {
    "docker.io": os.getenv("LINKOS_MIRROR_DOCKER", "docker.linkos.org"),
    "gcr.io": os.getenv("LINKOS_MIRROR_GCR", "gcr.linkos.org"),
    "ghcr.io": os.getenv("LINKOS_MIRROR_GHCR", "ghcr.linkos.org"),
    "quay.io": os.getenv("LINKOS_MIRROR_QUAY", "quay.linkos.org"),
    "registry.k8s.io": os.getenv("LINKOS_MIRROR_K8S", "k8s.linkos.org"),
    "k8s.gcr.io": os.getenv("LINKOS_MIRROR_K8S", "k8s.linkos.org"),
}

ROOTFS_DEVICE = os.getenv("ROOTFS_DEVICE", "/dev/sda")
DOCKER_LIMITS = {
    "cpu_period": 100000,
    "cpu_quota": 100000 * 6,
    "mem_limit": "12g",
    "memswap_limit": "12g",
    "pids_limit": 32768,
    "blkio_weight": 200,
    "device_read_bps": [{"Path": ROOTFS_DEVICE, "Rate": 30 * 1024 * 1024}],
    "device_write_bps": [{"Path": ROOTFS_DEVICE, "Rate": 30 * 1024 * 1024}],
    "device_read_iops": [{"Path": ROOTFS_DEVICE, "Rate": 2000}],
    "device_write_iops": [{"Path": ROOTFS_DEVICE, "Rate": 2000}],
    "dns": ["223.5.5.5"],
    "environment": ["TZ=Asia/Shanghai"],
}

### BEGIN copied from sweap


def sweap_get_docker_tag(row):
    uid = row["instance_id"]
    repo_name = row.get("repo", "")
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
    # All other repos: strip -vnan suffix
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]

    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]
    return tag


### END copied from sweap


def ts():
    return datetime.datetime.now().isoformat()


def fix_owner_and_mode(path: Path, mode: int | None = None):
    if not path.exists():
        return
    os.chown(path, uid=FILE_OWNER_UID, gid=FILE_OWNER_UID, follow_symlinks=False)
    if mode is not None:
        os.chmod(path, mode, follow_symlinks=False)


def has_existing_results(path: Path) -> bool:
    """Return True if result dir contains previous evaluation artifacts."""
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    for child in path.iterdir():
        if child.name == "run_main.log":
            continue
        return True
    return False


def cleanup_result_path_for_overwrite(path: Path):
    """Cleanup old run results but keep run_main.log if present."""
    if not path.is_dir():
        raise FileExistsError(f"result path exists but is not directory: {path}")

    for child in path.iterdir():
        if child.name == "run_main.log":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except FileNotFoundError:
                pass


def snapshot_run_config_files(result_path: Path, projects_file: Path, candidates_file: Path):
    result_path.mkdir(parents=True, exist_ok=True)

    for src in [projects_file, candidates_file]:
        dst = result_path / src.name
        shutil.copy2(src, dst)
        fix_owner_and_mode(dst, 0o644)

    if VERBOSE >= 1:
        print(
            f"{ts()} | saved run config files: {projects_file.name}, {candidates_file.name} -> {result_path}"
        )


# def read_patch_text_lossless(path: Path) -> str:
#     # Read patch bytes and decode without whitespace normalization.
#     # This preserves tabs/leading spaces exactly as written in patch.diff.
#     return path.read_bytes().decode("utf-8", errors="surrogateescape")


sweap_df = pd.read_csv("external_hf_v2.csv", engine="c", on_bad_lines="skip")
sweap_df = sweap_df.set_index("instance_id", drop=False)


class Instance:
    DOCKER_IMAGE_BASE: str = os.environ.get("DOCKER_IMAGE_BASE", "jefzda/sweap-images")

    def __init__(self, instance_id: str, ident: int):
        self.instance_id: str = instance_id
        self.ident: str = ident  # used as llm key name
        self.instance: pd.Series = sweap_df.loc[instance_id]
        self.env_docker_image: str = (
            f"{self.DOCKER_IMAGE_BASE}:{sweap_get_docker_tag(self.instance)}"
        )
        self.instance_input: dict[str, str] = {
            k: self.instance[k]
            for k in [
                "repo",
                "repo_language",
                "problem_statement",
                "requirements",
                "interface",
            ]
        }


@dataclass
class Candidate:
    run_name: str  # used as llm user name and output folder name
    agent_docker_image: str
    llm_quota_total: float
    llm_quota_instance: float
    enable_memory: bool
    timeout_s: float


class Workdir:
    def __init__(
        self, stem: str, cleanup_fn: Callable[["Workdir"], None] | None = None
    ):
        self.name: str = f"{stem}--{random.randint(0, 1000000)}"
        self.path: Path = WORK_PATH / self.name
        self.cleanup_fn: Callable[["Workdir"], None] | None = cleanup_fn

        self.path.mkdir(parents=True, exist_ok=True)

        if VERBOSE >= 1:
            print(f"{ts()} | create workdir {self.name}")

    def cleanup(self):
        if self.path:
            if VERBOSE >= 1:
                print(f"{ts()} | cleanup workdir {self.name}")

            if self.cleanup_fn:
                self.cleanup_fn(self)

            if self.path.exists():
                shutil.rmtree(self.path, ignore_errors=True)

            self.path = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        self.cleanup()


docker_client = docker.from_env()


def _split_registry(image: str) -> tuple[str | None, str]:
    first, sep, rest = image.partition("/")
    if sep and ("." in first or ":" in first or first == "localhost"):
        return first, rest
    return None, image


def _ensure_library_namespace(name_with_ref: str) -> str:
    if "@" in name_with_ref:
        repo, digest = name_with_ref.split("@", 1)
        ref = f"@{digest}"
    elif ":" in name_with_ref:
        repo, maybe_tag = name_with_ref.rsplit(":", 1)
        if "/" in maybe_tag:
            repo = name_with_ref
            ref = ""
        else:
            ref = f":{maybe_tag}"
    else:
        repo = name_with_ref
        ref = ""

    if "/" not in repo:
        repo = f"library/{repo}"
    return f"{repo}{ref}"


def _to_linkos_mirror(image: str) -> str | None:
    registry, rest = _split_registry(image)
    source_registry = registry or "docker.io"
    mirror_registry = LINKOS_MIRROR_MAP.get(source_registry)
    if not mirror_registry:
        return None

    if source_registry == "docker.io":
        rest = _ensure_library_namespace(rest)
    return f"{mirror_registry}/{rest}"


def pull_image_with_fallback(image: str):
    print(f"pulling {image}")
    mirror_image = _to_linkos_mirror(image) if ENABLE_LINKOS_MIRROR else None

    if mirror_image:
        try:
            if VERBOSE >= 1:
                print(f"{ts()} | trying mirror pull {mirror_image}")
            docker_client.images.pull(mirror_image)
            docker_client.images.get(mirror_image).tag(image)
            if VERBOSE >= 1:
                print(f"{ts()} | mirror pull succeeded: {mirror_image} -> {image}")
            return
        except Exception as mirror_err:
            if VERBOSE >= 1:
                print(
                    f"{ts()} | mirror pull failed for {mirror_image}: {repr(mirror_err)}"
                )
                print(f"{ts()} | fallback to direct pull {image}")

    docker_client.images.pull(image)


def remove_container_if_exists(container_name: str):
    try:
        stale = docker_client.containers.get(container_name)
    except docker.errors.NotFound:
        return

    if VERBOSE >= 1:
        print(f"{ts()} | remove stale container {container_name}")

    try:
        stale.stop(timeout=3)
    except Exception:
        pass

    stale.remove(force=True)


def collect_candidate_results(projects: list[list[str]], candidate: Candidate):
    rows: list[dict[str, str]] = []
    candidate_dir = RESULT_PATH / candidate.run_name
    output_path = RESULT_PATH / f"{candidate.run_name}.json"

    for pidx, project in enumerate(projects):
        for iidx, instance_id in enumerate(project):
            ident = f"p{pidx:02d}i{iidx:02d}"
            patch_path = candidate_dir / ident / "patch.diff"
            patch_text = ""

            if patch_path.is_file():
                patch_text = patch_path.read_text(errors="replace")
            else:
                if VERBOSE >= 1:
                    print(
                        f"{ts()} | {candidate.run_name} {ident}: patch.diff not found"
                    )

            rows.append(
                {
                    "instance_id": instance_id,
                    "patch": patch_text,
                    "prefix": candidate.run_name,
                }
            )

    with output_path.open("w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write("\n")

    fix_owner_and_mode(output_path, 0o644)
    if VERBOSE >= 1:
        print(
            f"{ts()} | {candidate.run_name}: collected {len(rows)} rows -> {output_path}"
        )


def eval_single_project(instances: list[Instance], candidate: Candidate):
    assert len(instances) > 0
    mem_dir = Workdir(f"{candidate.run_name}-mem-global-{instances[0].ident}")

    @contextmanager
    def llm_key_scope(key_alias):
        _create_key_resp = requests.get(
            f"{LITELLM_CONTROLLER_BASEURL}/harness/create_key",
            params={
                "user_id": candidate.run_name,
                "key_alias": key_alias,
                "quota": candidate.llm_quota_instance,
            },
        ).json()
        llm_key = _create_key_resp["key"]
        llm_keyhash = _create_key_resp["hash"]
        try:
            yield llm_key, llm_keyhash
        finally:
            if VERBOSE >= 1:
                print(f"{ts()} | cleanup llm key {llm_key} ({llm_keyhash})")

            requests.get(
                f"{LITELLM_CONTROLLER_BASEURL}/harness/delete_key",
                params={"key": llm_key},
            )

    @contextmanager
    def sshbox_scope(instance: Instance, pcap_path: Path):
        ssh_password = secrets.token_hex(10)
        sshbox_name = f"memcomp-{candidate.run_name}-{instance.ident}-sshbox"
        remove_container_if_exists(sshbox_name)
        cont = docker_client.containers.create(
            instance.env_docker_image,
            [ssh_password],
            entrypoint=["/mnt/sshbox/start.sh"],
            name=sshbox_name,
            detach=True,
            restart_policy={"Name": "no"},
            network="infra",
            volumes={
                str(SSHBOX_SCRIPT_PATH.resolve()): {
                    "bind": "/mnt/sshbox",
                    "mode": "ro",
                },
                str(pcap_path.resolve()): {"bind": "/mnt/pcap", "mode": "rw"},
            },
            **DOCKER_LIMITS,
        )
        docker_client.networks.get("internet").connect(cont)

        try:
            cont.start()

            cont.reload()
            sshbox_ip = cont.attrs["NetworkSettings"]["Networks"]["infra"]["IPAddress"]
            assert sshbox_ip
            sshbox_conn_str = f"root:{ssh_password}@{sshbox_ip}"

            yield sshbox_conn_str
        finally:
            if VERBOSE >= 1:
                print(f"{ts()} | cleanup sshbox {cont.name}")

            try:
                cont.stop(timeout=3)
                cont.remove(force=True)
            except Exception:
                if VERBOSE >= 1:
                    print(f"{ts()} | cleanup sshbox {cont.name} failed, try again")

                try:
                    cont.remove(force=True)
                except Exception:
                    if VERBOSE >= 1:
                        print(f"{ts()} | cleanup sshbox {cont.name} failed, ignoring")

    @contextmanager
    def agent_scope(
        inst_path: Path, mem_path: Path, llm_key: str, ssh_conn_str: str, log_path: Path
    ):
        agent_name = f"memcomp-{candidate.run_name}-{instance.ident}-agent"
        remove_container_if_exists(agent_name)
        cont = docker_client.containers.create(
            candidate.agent_docker_image,
            [
                "--instance-path",
                "/mnt/instance",
                "--memory-path",
                "/mnt/memory",
                "--llm-base-url",
                LITELLM_BASEURL_IN_DOCKER,
                "--llm-api-key",
                llm_key,
                "--env-ssh",
                ssh_conn_str,
            ],
            name=agent_name,
            detach=True,
            restart_policy={"Name": "no"},
            network="infra",
            volumes={
                str(inst_path.resolve()): {"bind": "/mnt/instance", "mode": "rw"},
                str(mem_path.resolve()): {"bind": "/mnt/memory", "mode": "rw"},
            },
            **DOCKER_LIMITS,
        )

        def log_thread():
            with log_path.open("wb") as logf:
                for content in cont.logs(stream=True, timestamps=True, follow=True):
                    logf.write(content)
                    logf.flush()

                    if VERBOSE >= 2:
                        t = ts()
                        for line in content.decode().splitlines():
                            line = line.partition(" ")[2]  # remove timestamp prefix
                            print(
                                f"{t} | {candidate.run_name} {instance.ident}: > {line}"
                            )

        try:
            cont.start()

            time.sleep(
                1
            )  # xxx: must start log_thread after container is started to properly receive logs
            log_thread = threading.Thread(target=log_thread, daemon=True)
            log_thread.start()

            yield cont
        finally:
            if VERBOSE >= 1:
                print(
                    f"{ts()} | cleanup agent {cont.name} ({candidate.agent_docker_image})"
                )

            try:
                cont.stop(timeout=3)
                cont.remove(force=True)
            except Exception:
                if VERBOSE >= 1:
                    print(
                        f"{ts()} | cleanup agent {cont.name} ({candidate.agent_docker_image}) failed, try again"
                    )

                try:
                    cont.remove(force=True)
                except Exception:
                    if VERBOSE >= 1:
                        print(
                            f"{ts()} | cleanup agent {cont.name} ({candidate.agent_docker_image}) failed, ignoring"
                        )
            log_thread.join()

    try:
        for instance in instances:
            print(f"{ts()} | {candidate.run_name} {instance.ident}: INIT")

            # create memory path
            if not candidate.enable_memory:
                mem_dir.cleanup()
                mem_dir = Workdir(f"{candidate.run_name}-{instance.ident}-mem")

            # create instance path
            inst_path = RESULT_PATH / candidate.run_name / instance.ident
            if inst_path.exists():
                shutil.rmtree(inst_path, ignore_errors=True)
            inst_path.mkdir(parents=True)
            fix_owner_and_mode(RESULT_PATH, 0o755)
            fix_owner_and_mode(inst_path.parent, 0o755)
            fix_owner_and_mode(inst_path, 0o755)

            def perform_cleanup(log_dir: Workdir):
                # collect memory snapshot

                shutil.copytree(mem_dir.path, log_dir.path / "memory")

                # move log_dir to results before it is cleaned up

                log_dest = inst_path / "_harness"
                if log_dest.is_dir():
                    shutil.rmtree(log_dest, ignore_errors=True)
                elif log_dest.is_file():
                    log_dest.unlink()

                shutil.move(log_dir.path, log_dest)

                # fix permission

                for dirpath, dirnames, filenames in os.walk(inst_path):
                    os.chown(
                        dirpath,
                        uid=FILE_OWNER_UID,
                        gid=FILE_OWNER_UID,
                        follow_symlinks=False,
                    )
                    os.chmod(dirpath, 0o755, follow_symlinks=False)
                    for filename in filenames:
                        fn = os.path.join(dirpath, filename)
                        os.chown(
                            fn,
                            uid=FILE_OWNER_UID,
                            gid=FILE_OWNER_UID,
                            follow_symlinks=False,
                        )
                        os.chmod(fn, 0o644, follow_symlinks=False)

                # fix upper-level directories so moving run folders does not require root.
                fix_owner_and_mode(inst_path.parent, 0o755)
                fix_owner_and_mode(RESULT_PATH, 0o755)

            # create log path
            with Workdir(
                f"{candidate.run_name}-{instance.ident}-log", cleanup_fn=perform_cleanup
            ) as log_dir:
                llm_keyhash = "???"

                with (
                    # create llm key and sshbox
                    llm_key_scope(instance.ident) as (llm_key, llm_keyhash_),
                    sshbox_scope(instance, log_dir.path / "pcap") as sshbox_conn_str,
                    # create harness system log file
                    (log_dir.path / "system.log").open("w") as logf,
                ):
                    llm_keyhash = llm_keyhash_

                    logf.write(f"candidate: {asdict(candidate)}\n")
                    logf.write(
                        f"instance: {instance.ident}, instance_id = {instance.instance_id}\n"
                    )
                    logf.write(f"llm key: {llm_key}, hash = {llm_keyhash}\n")
                    logf.write(f"sshbox: {sshbox_conn_str}\n")

                    if VERBOSE >= 1:
                        print(
                            f"{ts()} | {candidate.run_name} {instance.ident}: key = {llm_key} ; hash = {llm_keyhash} ; ssh = {sshbox_conn_str}"
                        )

                    # write input

                    with (inst_path / "instance.json").open("w") as f:
                        json.dump(instance.instance_input, f, indent=4)

                    # run candidate agent

                    time.sleep(1)

                    logf.write(
                        f'load: {psutil.getloadavg()} with {psutil.cpu_count()} cores\ncpu: {psutil.cpu_times_percent()}\nmem: {psutil.virtual_memory()}\ndisk: {psutil.disk_usage(".")}\n'
                    )

                    print(f"{ts()} | {candidate.run_name} {instance.ident}: START")
                    logf.write(f"start running agent @ {ts()}\n")

                    with agent_scope(
                        inst_path,
                        mem_dir.path,
                        llm_key,
                        sshbox_conn_str,
                        log_dir.path / "agent.log",
                    ) as cont:
                        try:
                            agent_ret = cont.wait(timeout=candidate.timeout_s)
                        except Exception as e:
                            if VERBOSE >= 1:
                                print(
                                    f"{ts()} | {candidate.run_name} {instance.ident}: agent timeout: {repr(e)}"
                                )

                            logf.write(f"agent timeout:\n{traceback.format_exc()}\n")
                        else:
                            if VERBOSE >= 1:
                                print(
                                    f"{ts()} | {candidate.run_name} {instance.ident}: agent ret = {agent_ret}"
                                )

                            logf.write(
                                f'agent finished: retcode = {agent_ret["StatusCode"]}\n'
                            )

                        logf.write(f"stop running agent @ {ts()}\n")

                    print(f"{ts()} | {candidate.run_name} {instance.ident}: FIN")
                    logf.write(f"agent removed @ {ts()}\n")

                    # query llm usage

                    usage = requests.get(
                        f"{LITELLM_CONTROLLER_BASEURL}/harness/query_balance",
                        params={"user_id": candidate.run_name, "key": llm_key},
                    ).json()

                    if VERBOSE >= 1:
                        print(
                            f"{ts()} | {candidate.run_name} {instance.ident}: llm usage = {usage}"
                        )

                    logf.write(f"llm usage: {usage}\n")

                # when the context manager exits: llm key will be deleted, sshbox will be removed, logf will be flushed

                time.sleep(1)

                # collect litellm traj

                traj_src = LITELLM_TRAJ_PATH / llm_keyhash
                traj_dest = log_dir.path / "traj"

                if traj_src.is_dir():
                    shutil.move(traj_src, traj_dest)
                else:
                    if VERBOSE >= 1:
                        print(
                            f"{ts()} | {candidate.run_name} {instance.ident}: traj dir not found"
                        )

            # when the context manager exits: log_dir will be cleaned up

        print(f"{ts()} | {candidate.run_name}: DONE")

    except Exception:
        print(f"{ts()} | {candidate.run_name}: FATAL ERROR")
        traceback.print_exc()

    finally:
        mem_dir.cleanup()


def eval_candidate(projects: list[list[str]], candidate: Candidate):
    @contextmanager
    def llm_user_scope():
        res = requests.get(
            f"{LITELLM_CONTROLLER_BASEURL}/harness/create_user",
            params={
                "user_id": candidate.run_name,
                "user_alias": candidate.run_name,
                "quota": candidate.llm_quota_total,
            },
        )
        res.raise_for_status()
        try:
            yield
        finally:
            if VERBOSE >= 1:
                print(f"{ts()} | cleanup llm user {candidate.run_name}")

            res = requests.get(
                f"{LITELLM_CONTROLLER_BASEURL}/harness/delete_user",
                params={"user_id": candidate.run_name},
            )
            res.raise_for_status()

    # create llm user
    with llm_user_scope():
        workers = []

        # load instances
        for pidx, project in enumerate(projects):
            instances = []
            for iidx, instance_id in enumerate(project):
                instances.append(Instance(instance_id, f"p{pidx:02d}i{iidx:02d}"))

            # start eval process (not thread because finally statements in threads are broken)
            p = multiprocessing.Process(
                target=eval_single_project, args=(instances, candidate)
            )
            workers.append(p)
            p.start()

        # wait for eval to finish
        for worker in workers:
            worker.join()

    # when the context manager exits: llm user will be deleted
    collect_candidate_results(projects, candidate)


if __name__ == "__main__":
    projects_file = Path("projects_debug.json")
    candidates_file = Path("candidates.json")

    with projects_file.open() as f:
        projects = json.load(f)
    with candidates_file.open() as f:
        candidates = json.load(f)

    print(
        f"loaded candidates = {len(candidates)} ; instances = {sum(len(p) for p in projects)} = {[len(p) for p in projects]}"
    )

    def san_check():
        # check path

        assert SSHBOX_SCRIPT_PATH.exists()
        assert LITELLM_TRAJ_PATH.exists()

        if RESULT_PATH.exists() and has_existing_results(RESULT_PATH):
            print(f"- {RESULT_PATH}")
            while True:
                ans = (
                    input(f"result path already exists, overwrite it? [y/n]: ")
                    .strip()
                    .lower()
                )
                # Keep terminal/log output aligned when stdout is piped through tee.
                print("", flush=True)
                if ans in {"y", "yes"}:
                    print(
                        "WARNING: will cleanup existing results in 10 seconds, CTRL+C NOW to abort",
                        flush=True,
                    )
                    time.sleep(10)
                    print(f"{ts()} | cleanup old results: {RESULT_PATH}", flush=True)
                    cleanup_result_path_for_overwrite(RESULT_PATH)
                    print(f"{ts()} | cleanup finished: {RESULT_PATH}", flush=True)
                    break
                if ans in {"n", "no"}:
                    raise SystemExit("aborted by user")
                print("invalid input, please enter y or n")

        RESULT_PATH.mkdir(parents=True, exist_ok=True)
        WORK_PATH.mkdir(parents=True, exist_ok=True)
        fix_owner_and_mode(RESULT_PATH, 0o755)
        fix_owner_and_mode(WORK_PATH, 0o755)

        # check url

        res = requests.get(f"{LITELLM_CONTROLLER_BASEURL}/harness/health")
        res.raise_for_status()
        assert "v1" in res.json()["compat"]

        cont = docker_client.containers.run(
            "curlimages/curl:8.6.0",
            [
                "-sS",
                "-m",
                "5",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                f"{LITELLM_BASEURL_IN_DOCKER}/health/liveliness",
            ],
            detach=True,
            network="infra",
            **DOCKER_LIMITS,
        )
        cont.wait()
        log = cont.logs().decode(errors="replace")
        cont.remove(force=True)
        assert log.strip() == "200"

        # check docker image

        images = []
        for c in candidates:
            images.append(c["agent_docker_image"])
        for p in projects:
            for i in p:
                images.append(
                    f"{Instance.DOCKER_IMAGE_BASE}:{sweap_get_docker_tag(sweap_df.loc[i])}"
                )

        for image in images:
            try:
                docker_client.images.get(image)
            except docker.errors.ImageNotFound:
                pull_image_with_fallback(image)

    if COLLECT_ONLY:
        if not RESULT_PATH.exists():
            raise FileNotFoundError(
                f"result path not found for collect-only mode: {RESULT_PATH}"
            )
        snapshot_run_config_files(RESULT_PATH, projects_file, candidates_file)

        print(f"{ts()} | begin collect-only")
        for c in candidates:
            collect_candidate_results(projects, Candidate(**c))
        print(f"{ts()} | done collect-only")
    else:
        san_check()
        snapshot_run_config_files(RESULT_PATH, projects_file, candidates_file)
        print(f"{ts()} | sanity check passed")

        print(f"{ts()} | begin")
        for c in candidates:
            eval_candidate(projects, Candidate(**c))
        print(f"{ts()} | done")
