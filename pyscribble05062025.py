# @title Setup Google Cloud project

# @markdown 1. [Make sure that billing is enabled for your project](https://cloud.google.com/billing/docs/how-to/modify-project).

# @markdown 2. **[Optional]** Set region. If left unchanged, the region defaults to us-east4 for using H200 GPUs.

REGION = "us-east1"  # @param {type:"string"}

# @markdown 3. If you want to run predictions with H100 GPUs, we recommend using the regions listed below. **NOTE:** Make sure you have associated quota in selected regions. Click the links to see your current quota for Spot VM H100 GPUs: [`CustomModelServingPreemptibleH100GPUsPerProjectPerRegion`](https://console.cloud.google.com/iam-admin/quotas?metric=aiplatform.googleapis.com%2Fcustom_model_serving_preemptible_nvidia_h100_gpus) and regular VM H100s: [`CustomModelServingH100GPUsPerProjectPerRegion`](https://console.cloud.google.com/iam-admin/quotas?metric=aiplatform.googleapis.com%2Fcustom_model_serving_nvidia_h100_gpus)..

# @markdown > | Machine Type | Accelerator Type | Recommended Regions |
# @markdown | ----------- | ----------- | ----------- |
# @markdown | a3-highgpu-8g (Spot VM) | 8 NVIDIA_H100_80GB | us-central1, europe-west4, asia-southeast1 |
# @markdown | a3-highgpu-8g (regular VM) | 8 NVIDIA_H100_80GB | us-central1, europe-west4, us-west1, asia-southeast1 |

# Upgrade Vertex AI SDK.

# Import the necessary packages
import importlib
import os
import time
from typing import Tuple

import requests
from google import auth
from google.cloud import aiplatform

# Upgrade Vertex AI SDK.

common_util = importlib.import_module(
    "vertex-ai-samples.community-content.vertex_model_garden.model_oss.notebook_util.common_util"
)


def check_quota(
    project_id: str,
    region: str,
    resource_id: str,
    accelerator_count: int,
):
    """Checks if the project and the region has the required quota."""
    quota = common_util.get_quota(project_id, region, resource_id)
    quota_request_instruction = (
        "Either use "
        "a different region or request additional quota. Follow "
        "instructions here "
        "https://cloud.google.com/docs/quotas/view-manage#requesting_higher_quota"
        " to check quota in a region or request additional quota for "
        "your project."
    )
    if quota == -1:
        raise ValueError(
            f"Quota not found for: {resource_id} in {region}."
            f" {quota_request_instruction}"
        )
    if quota < accelerator_count:
        raise ValueError(
            f"Quota not enough for {resource_id} in {region}: {quota} <"
            f" {accelerator_count}. {quota_request_instruction}"
        )


LABEL = "vllm_gpu"
models, endpoints = {}, {}

# Get the default cloud project id.
PROJECT_ID = "pyscribble"

# Get the default region for launching jobs.
if not REGION:
    REGION = "us-east1"

# Initialize Vertex AI API.
print("Initializing Vertex AI API.")
aiplatform.init(project=PROJECT_ID, location=REGION)


import vertexai

vertexai.init(
    project=PROJECT_ID,
    location=REGION,
)

# @title Set the model variants

# @markdown It's recommended to use the region selected by the deployment button on the model card. If the deployment button is not available, it's recommended to stay with the default region of the notebook.

# @markdown Multi-host GPU serving is a preview feature.

# @markdown Set the model to deploy.

base_model_name = "DeepSeek-R1"  # @param ["DeepSeek-V3", "DeepSeek-V3-Base", "DeepSeek-V3-0324", "DeepSeek-R1"] {isTemplate:true}
model_id = "deepseek-ai/" + base_model_name
hf_model_id = model_id
if "R1" in model_id:
    model_user_id = "deepseek-r1"
    model_id = f"gs://vertex-model-garden-restricted-us/{model_id}"
else:
    model_user_id = "deepseek-v3"

# fmt: off
PUBLISHER_MODEL_NAME = f"publishers/deepseek-ai/models/{model_user_id}@{base_model_name.lower()}"
# fmt: on

# @markdown Set use_dedicated_endpoint to False if you don't want to use [dedicated endpoint](https://cloud.google.com/vertex-ai/docs/general/deployment#create-dedicated-endpoint). Note that [dedicated endpoint does not support VPC Service Controls](https://cloud.google.com/vertex-ai/docs/predictions/choose-endpoint-type), uncheck the box if you are using VPC-SC.
use_dedicated_endpoint = True  # @param {type:"boolean"}



"""---------"""
# @title Deploy with customized configs

# @markdown This section uploads DeepSeek models to Model Registry and deploys them to a Vertex Prediction Endpoint. It takes ~1 hour to finish.

# @markdown The following vLLM container version has been validated. The version will be continuously updated to incorporate latest optimizations and features.
# The pre-built serving docker image for vLLM past v0.7.3, https://github.com/vllm-project/vllm/commit/f6bb18fd9a19e5e4fb1991339638fc666d06b27a.
VLLM_DOCKER_URI = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20250312_0916_RC01"

# @markdown Choose whether to use a [Spot VM](https://cloud.google.com/compute/docs/instances/spot) for the deployment.
is_spot = False  # @param {type:"boolean"}

# @markdown Find Vertex AI prediction supported accelerators and regions at https://cloud.google.com/vertex-ai/docs/predictions/configure-compute.
accelerator_type = "NVIDIA_H200_141GB"  # @param ["NVIDIA_H200_141GB", "NVIDIA_H100_80GB"] {isTemplate:true}
accelerator_count = 8
if accelerator_type == "NVIDIA_H200_141GB":
    machine_type = "a3-ultragpu-8g"
    multihost_gpu_node_count = 1
    if is_spot:
        raise ValueError("H200 GPUs are currently not available in Spot VM quota.")
    else:
        resource_id = "custom_model_serving_nvidia_h200_gpus"
else:
    machine_type = "a3-highgpu-8g"
    multihost_gpu_node_count = 2
    if is_spot:
        resource_id = "custom_model_serving_preemptible_nvidia_h100_gpus"
    else:
        resource_id = "custom_model_serving_nvidia_h100_gpus"

check_quota(
    project_id=PROJECT_ID,
    region=REGION,
    resource_id=resource_id,
    accelerator_count=int(accelerator_count * multihost_gpu_node_count),
)

if accelerator_type == "NVIDIA_H200_141GB":
    # @markdown With a single host of 8 x H200s, speculative decoding with MTP and a context length of 8192 are supported in the specified configuration. The configuration has been validated for stability and performance.
    pipeline_parallel_size = 1
    gpu_memory_utilization = 0.75
    max_model_len = 8192  # Maximum context length.
    enable_chunked_prefill = False
    max_num_seqs = 64
    kv_cache_dtype = "auto"
    num_speculative_tokens = 3
    speculative_draft_tensor_parallel_size = 8
else:
    # @markdown With 2 hosts of 8 x H100s, chunked prefill and a context length of 163840 are supported in the specified configuration. The configuration has been validated for stability and performance.
    pipeline_parallel_size = 2
    gpu_memory_utilization = 0.82
    max_model_len = 163840  # Maximum context length.
    enable_chunked_prefill = True
    max_num_seqs = 64
    kv_cache_dtype = "auto"
    num_speculative_tokens = None
    speculative_draft_tensor_parallel_size = None


# # The pre-built serving docker image and configuration for vLLM v0.7.2.
# VLLM_DOCKER_URI = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20250304_0916_RC01"
# accelerator_type = "NVIDIA_H100_80GB"
# accelerator_count = 8
# machine_type = "a3-highgpu-8g"
# multihost_gpu_node_count = 2
# pipeline_parallel_size = 2
# gpu_memory_utilization = 0.8
# max_model_len = 4096  # Maximum context length.
# enable_chunked_prefill = False
# max_num_seqs = 64
# kv_cache_dtype = "auto"
# num_speculative_tokens = None
# speculative_draft_tensor_parallel_size = None

# # The pre-built serving docker image and configuration for vLLM v0.6.6.post1.
# VLLM_DOCKER_URI = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20250130_0916_RC01"
# accelerator_type = "NVIDIA_H100_80GB"
# accelerator_count = 8
# machine_type = "a3-highgpu-8g"
# multihost_gpu_node_count = 2
# pipeline_parallel_size = 1
# gpu_memory_utilization = 0.9
# max_model_len = 32768  # Maximum context length.
# enable_chunked_prefill = False
# max_num_seqs = 128
# kv_cache_dtype = "fp8"
# num_speculative_tokens = None
# speculative_draft_tensor_parallel_size = None


# Enable automatic prefix caching using GPU HBM
enable_prefix_cache = False
# Setting this value >0 will use the idle host memory for a second-tier prefix kv
# cache beneath the HBM cache. It only has effect if enable_prefix_cache=True.
# The range of this value: [0, 1)
# Setting host_prefix_kv_cache_utilization_target to 0 will disable the host memory prefix kv cache.
host_prefix_kv_cache_utilization_target = 0

# @markdown To enable the auto-scaling in deployment, you can set the following options:

min_replica_count = 1  # @param {type:"integer"}
max_replica_count = 1  # @param {type:"integer"}
required_replica_count = 1  # @param {type:"integer"}

# @markdown Set the target of GPU duty cycle or CPU usage between 1 and 100 for auto-scaling.
autoscale_by_gpu_duty_cycle_target = 0  # @param {type:"integer"}
autoscale_by_cpu_usage_target = 0  # @param {type:"integer"}

# @markdown Note: GPU duty cycle is not the most accurate metric for scaling workloads. More advanced auto-scaling metrics are coming soon. See [the public doc](https://cloud.google.com/vertex-ai/docs/predictions/configure-compute#autoscaling) for more details.


def deploy_model_vllm_multihost_spec_decode(
    model_name: str,
    model_id: str,
    publisher: str,
    publisher_model_id: str,
    service_account: str = None,
    base_model_id: str = None,
    machine_type: str = "g2-standard-8",
    accelerator_type: str = "NVIDIA_L4",
    accelerator_count: int = 1,
    multihost_gpu_node_count: int = 1,
    pipeline_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    dtype: str = "auto",
    kv_cache_dtype: str = "auto",
    enable_trust_remote_code: bool = False,
    enforce_eager: bool = False,
    enable_lora: bool = False,
    enable_chunked_prefill: bool = False,
    enable_prefix_cache: bool = False,
    host_prefix_kv_cache_utilization_target: float = 0.0,
    max_loras: int = 1,
    max_cpu_loras: int = 8,
    use_dedicated_endpoint: bool = False,
    max_num_seqs: int = 256,
    num_speculative_tokens: int = None,
    speculative_draft_tensor_parallel_size: int = None,
    model_type: str = None,
    min_replica_count: int = 1,
    max_replica_count: int = 1,
    required_replica_count: int = 1,
    autoscale_by_gpu_duty_cycle_target: int = 0,
    autoscale_by_cpu_usage_target: int = 0,
    is_spot: bool = True,
) -> Tuple[aiplatform.Model, aiplatform.Endpoint]:
    """Deploys trained models with vLLM into Vertex AI."""
    endpoint = aiplatform.Endpoint.create(
        display_name=f"{model_name}-endpoint",
        dedicated_endpoint_enabled=use_dedicated_endpoint,
    )

    if not base_model_id:
        base_model_id = model_id

    # See https://docs.vllm.ai/en/latest/models/engine_args.html for a list of possible arguments with descriptions.
    vllm_args = [
        "python",
        "-m",
        "vllm.entrypoints.api_server",
        "--host=0.0.0.0",
        "--port=8080",
        f"--model={model_id}",
        f"--tensor-parallel-size={int(accelerator_count * multihost_gpu_node_count / pipeline_parallel_size)}",
        f"--pipeline-parallel-size={pipeline_parallel_size}",
        "--swap-space=16",
        f"--gpu-memory-utilization={gpu_memory_utilization}",
        f"--max-model-len={max_model_len}",
        f"--dtype={dtype}",
        f"--kv-cache-dtype={kv_cache_dtype}",
        f"--max-loras={max_loras}",
        f"--max-cpu-loras={max_cpu_loras}",
        f"--max-num-seqs={max_num_seqs}",
        "--disable-log-requests",
    ]

    if multihost_gpu_node_count > 1:
        vllm_args = ["/vllm-workspace/ray_launcher.sh"] + vllm_args

    if enable_trust_remote_code:
        vllm_args.append("--trust-remote-code")

    if enforce_eager:
        vllm_args.append("--enforce-eager")

    if enable_lora:
        vllm_args.append("--enable-lora")

    if enable_chunked_prefill:
        vllm_args.append("--enable-chunked-prefill")

    if enable_prefix_cache:
        vllm_args.append("--enable-prefix-caching")

    if 0 < host_prefix_kv_cache_utilization_target < 1:
        vllm_args.append(
            f"--host-prefix-kv-cache-utilization-target={host_prefix_kv_cache_utilization_target}"
        )

    if num_speculative_tokens is not None:
        vllm_args.append(f"--num-speculative-tokens={num_speculative_tokens}")

    if speculative_draft_tensor_parallel_size is not None:
        vllm_args.append(
            f"--speculative-draft-tensor-parallel-size={speculative_draft_tensor_parallel_size}"
        )

    if model_type:
        vllm_args.append(f"--model-type={model_type}")

    env_vars = {
        "MODEL_ID": base_model_id,
        "DEPLOY_SOURCE": "notebook",
    }

    # HF_TOKEN is not a compulsory field and may not be defined.
    try:
        if HF_TOKEN:
            env_vars["HF_TOKEN"] = HF_TOKEN
    except NameError:
        pass

    model = aiplatform.Model.upload(
        display_name=model_name,
        serving_container_image_uri=VLLM_DOCKER_URI,
        serving_container_args=vllm_args,
        serving_container_ports=[8080],
        serving_container_predict_route="/generate",
        serving_container_health_route="/ping",
        serving_container_environment_variables=env_vars,
        serving_container_shared_memory_size_mb=(16 * 1024),  # 16 GB
        serving_container_deployment_timeout=7200,
        model_garden_source_model_name=(
            f"publishers/{publisher}/models/{publisher_model_id}"
        ),
    )
    print(
        f"Deploying {model_name} on {machine_type} with {int(accelerator_count * multihost_gpu_node_count)} {accelerator_type} GPU(s)."
    )

    creds, _ = auth.default()
    auth_req = auth.transport.requests.Request()
    creds.refresh(auth_req)

    url = f"https://{REGION}-aiplatform.googleapis.com/ui/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{endpoint.name}:deployModel"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {creds.token}",
    }
    data = {
        "deployedModel": {
            "model": model.resource_name,
            "displayName": model_name,
            "dedicatedResources": {
                "machineSpec": {
                    "machineType": machine_type,
                    "multihostGpuNodeCount": multihost_gpu_node_count,
                    "acceleratorType": accelerator_type,
                    "acceleratorCount": accelerator_count,
                },
                "minReplicaCount": min_replica_count,
                "requiredReplicaCount": required_replica_count,
                "maxReplicaCount": max_replica_count,
            },
            "system_labels": {
                "NOTEBOOK_NAME": "model_garden_pytorch_deepseek_deployment.ipynb",
                "NOTEBOOK_ENVIRONMENT": common_util.get_deploy_source(),
            },
        },
    }
    if service_account:
        data["deployedModel"]["serviceAccount"] = service_account
    if is_spot:
        data["deployedModel"]["dedicatedResources"]["spot"] = True
    if autoscale_by_gpu_duty_cycle_target > 0 or autoscale_by_cpu_usage_target > 0:
        data["deployedModel"]["dedicatedResources"]["autoscalingMetricSpecs"] = []
        if autoscale_by_gpu_duty_cycle_target > 0:
            data["deployedModel"]["dedicatedResources"][
                "autoscalingMetricSpecs"
            ].append(
                {
                    "metricName": "aiplatform.googleapis.com/prediction/online/accelerator/duty_cycle",
                    "target": autoscale_by_gpu_duty_cycle_target,
                }
            )
        if autoscale_by_cpu_usage_target > 0:
            data["deployedModel"]["dedicatedResources"][
                "autoscalingMetricSpecs"
            ].append(
                {
                    "metricName": "aiplatform.googleapis.com/prediction/online/cpu/utilization",
                    "target": autoscale_by_cpu_usage_target,
                }
            )
    response = requests.post(url, headers=headers, json=data)
    print(f"Deploy Model response: {response.json()}")
    if response.status_code != 200 or "name" not in response.json():
        raise ValueError(f"Failed to deploy model: {response.text}")
    common_util.poll_and_wait(response.json()["name"], REGION, 7200)
    print("endpoint_name:", endpoint.name)

    return model, endpoint


models["vllm_gpu"], endpoints["vllm_gpu"] = deploy_model_vllm_multihost_spec_decode(
    model_name=common_util.get_job_name_with_datetime(prefix="deepseek-serve"),
    model_id=model_id,
    publisher="deepseek-ai",
    publisher_model_id=("deepseek-v3" if "V3" in model_id else "deepseek-r1"),
    base_model_id=hf_model_id,
    machine_type=machine_type,
    accelerator_type=accelerator_type,
    accelerator_count=accelerator_count,
    multihost_gpu_node_count=multihost_gpu_node_count,
    pipeline_parallel_size=pipeline_parallel_size,
    gpu_memory_utilization=gpu_memory_utilization,
    max_model_len=max_model_len,
    max_num_seqs=max_num_seqs,
    kv_cache_dtype=kv_cache_dtype,
    enable_trust_remote_code=True,
    enforce_eager=False,
    enable_lora=False,
    enable_chunked_prefill=enable_chunked_prefill,
    num_speculative_tokens=num_speculative_tokens,
    speculative_draft_tensor_parallel_size=speculative_draft_tensor_parallel_size,
    enable_prefix_cache=enable_prefix_cache,
    host_prefix_kv_cache_utilization_target=host_prefix_kv_cache_utilization_target,
    use_dedicated_endpoint=use_dedicated_endpoint,
    min_replica_count=min_replica_count,
    max_replica_count=max_replica_count,
    required_replica_count=required_replica_count,
    autoscale_by_gpu_duty_cycle_target=autoscale_by_gpu_duty_cycle_target,
    autoscale_by_cpu_usage_target=autoscale_by_cpu_usage_target,
    is_spot=is_spot,
)
# @markdown Click "Show Code" to see more details.
