// Hardware and WebGPU capability inspector.
// ASD-STE100 guidelines: short sentences, active voice, simple terms.

export async function getHardwareInfo() {
  const info = {
    webgpuSupported: false,
    adapterName: "Unavailable",
    vendor: "Unknown",
    architecture: "Unknown",
    driver: "Unknown",
    deviceMemoryGB: navigator.deviceMemory || "Unknown",
    hardwareConcurrency: navigator.hardwareConcurrency || "Unknown",
    maxBufferSizeMB: 0,
    maxStorageBufferMB: 0,
    maxComputeWorkgroupStorageKB: 0,
    limits: {}
  };

  if (!navigator.gpu) {
    return info;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      info.webgpuSupported = false;
      info.adapterName = "No WebGPU adapter found";
      return info;
    }

    info.webgpuSupported = true;

    // Read adapter info if available
    if (adapter.info) {
      info.adapterName = adapter.info.description || adapter.info.device || "Generic WebGPU Adapter";
      info.vendor = adapter.info.vendor || "Unknown";
      info.architecture = adapter.info.architecture || "Unknown";
      info.driver = adapter.info.driver || "Unknown";
    } else if (typeof adapter.requestAdapterInfo === "function") {
      const aInfo = await adapter.requestAdapterInfo();
      info.adapterName = aInfo.description || aInfo.device || "Generic WebGPU Adapter";
      info.vendor = aInfo.vendor || "Unknown";
      info.architecture = aInfo.architecture || "Unknown";
      info.driver = aInfo.driver || "Unknown";
    }

    // Extract key buffer and workgroup limits
    const limits = adapter.limits;
    if (limits) {
      info.maxBufferSizeMB = Math.round(limits.maxBufferSize / (1024 * 1024));
      info.maxStorageBufferMB = Math.round(limits.maxStorageBufferBindingSize / (1024 * 1024));
      info.maxComputeWorkgroupStorageKB = Math.round(limits.maxComputeWorkgroupStorageSize / 1024);
      info.limits = {
        maxBufferSize: limits.maxBufferSize,
        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup
      };
    }
  } catch (error) {
    console.error("Failed to query WebGPU adapter:", error);
    info.webgpuSupported = false;
    info.adapterName = "Error querying adapter: " + error.message;
  }

  return info;
}
