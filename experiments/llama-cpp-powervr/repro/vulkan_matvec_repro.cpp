#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <vector>

static void vk_check(VkResult r, const char * what) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed: " + std::to_string(r));
    }
}

static std::vector<uint32_t> read_spv(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("failed to open SPIR-V file");
    }
    const std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint32_t> data((size + 3) / 4);
    if (!f.read(reinterpret_cast<char *>(data.data()), size)) {
        throw std::runtime_error("failed to read SPIR-V file");
    }
    return data;
}

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void * mapped = nullptr;
    VkDeviceSize size = 0;
};

static uint32_t find_memory_type(VkPhysicalDevice physical, uint32_t bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props {};
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    throw std::runtime_error("no suitable memory type");
}

static Buffer make_buffer(VkPhysicalDevice physical, VkDevice device, VkDeviceSize size) {
    Buffer out;
    out.size = size;
    VkBufferCreateInfo bi { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(device, &bi, nullptr, &out.buffer), "vkCreateBuffer");

    VkMemoryRequirements req {};
    vkGetBufferMemoryRequirements(device, out.buffer, &req);
    VkMemoryAllocateInfo ai { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(physical, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vk_check(vkAllocateMemory(device, &ai, nullptr, &out.memory), "vkAllocateMemory");
    vk_check(vkBindBufferMemory(device, out.buffer, out.memory, 0), "vkBindBufferMemory");
    vk_check(vkMapMemory(device, out.memory, 0, size, 0, &out.mapped), "vkMapMemory");
    return out;
}

int main(int argc, char ** argv) {
    try {
        const uint32_t rows = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 1024;
        const uint32_t cols = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 1024;
        const char * spv_path = argc > 3 ? argv[3] : "matvec_scalar.spv";

        VkApplicationInfo app { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        app.pApplicationName = "powervr-matvec-repro";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        ici.pApplicationInfo = &app;
        VkInstance instance;
        vk_check(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance");

        uint32_t physical_count = 0;
        vk_check(vkEnumeratePhysicalDevices(instance, &physical_count, nullptr), "vkEnumeratePhysicalDevices");
        if (physical_count == 0) {
            throw std::runtime_error("no Vulkan physical devices");
        }
        std::vector<VkPhysicalDevice> physicals(physical_count);
        vk_check(vkEnumeratePhysicalDevices(instance, &physical_count, physicals.data()), "vkEnumeratePhysicalDevices");
        VkPhysicalDevice physical = physicals[0];

        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_family_count, qprops.data());
        uint32_t queue_family = UINT32_MAX;
        for (uint32_t i = 0; i < queue_family_count; ++i) {
            if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queue_family = i;
                break;
            }
        }
        if (queue_family == UINT32_MAX) {
            throw std::runtime_error("no compute queue");
        }

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qci.queueFamilyIndex = queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        VkDevice device;
        vk_check(vkCreateDevice(physical, &dci, nullptr, &device), "vkCreateDevice");
        VkQueue queue;
        vkGetDeviceQueue(device, queue_family, 0, &queue);

        Buffer a = make_buffer(physical, device, VkDeviceSize(rows) * cols * sizeof(float));
        Buffer b = make_buffer(physical, device, VkDeviceSize(cols) * sizeof(float));
        Buffer d = make_buffer(physical, device, VkDeviceSize(rows) * sizeof(float));
        Buffer fuse0 = make_buffer(physical, device, sizeof(float));
        Buffer fuse1 = make_buffer(physical, device, sizeof(float));

        float * pa = static_cast<float *>(a.mapped);
        float * pb = static_cast<float *>(b.mapped);
        float * pd = static_cast<float *>(d.mapped);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                const int v = int((r * 17 + c * 13) % 97) - 48;
                pa[uint64_t(r) * cols + c] = float(v) / 97.0f;
            }
        }
        for (uint32_t c = 0; c < cols; ++c) {
            const int v = int((c * 19) % 89) - 44;
            pb[c] = float(v) / 89.0f;
        }
        for (uint32_t r = 0; r < rows; ++r) {
            pd[r] = 0.0f;
        }
        *static_cast<float *>(fuse0.mapped) = 0.0f;
        *static_cast<float *>(fuse1.mapped) = 0.0f;

        VkDescriptorSetLayoutBinding bindings[5] {};
        for (uint32_t i = 0; i < 5; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        dlci.bindingCount = 5;
        dlci.pBindings = bindings;
        VkDescriptorSetLayout dsl;
        vk_check(vkCreateDescriptorSetLayout(device, &dlci, nullptr, &dsl), "vkCreateDescriptorSetLayout");

        VkPushConstantRange pcr {};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = 13 * sizeof(uint32_t);
        VkPipelineLayoutCreateInfo plci { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VkPipelineLayout layout;
        vk_check(vkCreatePipelineLayout(device, &plci, nullptr, &layout), "vkCreatePipelineLayout");

        std::vector<uint32_t> spv = read_spv(spv_path);
        VkShaderModuleCreateInfo smci { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smci.codeSize = spv.size() * sizeof(uint32_t);
        smci.pCode = spv.data();
        VkShaderModule shader;
        vk_check(vkCreateShaderModule(device, &smci, nullptr, &shader), "vkCreateShaderModule");

        VkPipelineShaderStageCreateInfo stage { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";
        uint32_t spec_data[3] { 1, 1, 1 };
        VkSpecializationMapEntry spec_entries[3] {};
        for (uint32_t i = 0; i < 3; ++i) {
            spec_entries[i].constantID = i;
            spec_entries[i].offset = i * sizeof(uint32_t);
            spec_entries[i].size = sizeof(uint32_t);
        }
        VkSpecializationInfo spec_info {};
        spec_info.mapEntryCount = 3;
        spec_info.pMapEntries = spec_entries;
        spec_info.dataSize = sizeof(spec_data);
        spec_info.pData = spec_data;
        stage.pSpecializationInfo = &spec_info;
        VkComputePipelineCreateInfo cpci { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        cpci.stage = stage;
        cpci.layout = layout;
        VkPipeline pipeline;
        vk_check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline), "vkCreateComputePipelines");

        const uint32_t chunk_rows = 256;
        const uint32_t chunk_count = (rows + chunk_rows - 1) / chunk_rows;

        VkDescriptorPoolSize pool_size {};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 5 * chunk_count;
        VkDescriptorPoolCreateInfo dpci { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        dpci.maxSets = chunk_count;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &pool_size;
        VkDescriptorPool pool;
        vk_check(vkCreateDescriptorPool(device, &dpci, nullptr, &pool), "vkCreateDescriptorPool");

        std::vector<VkDescriptorSetLayout> layouts(chunk_count, dsl);
        std::vector<VkDescriptorSet> sets(chunk_count);
        VkDescriptorSetAllocateInfo dsai { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dsai.descriptorPool = pool;
        dsai.descriptorSetCount = chunk_count;
        dsai.pSetLayouts = layouts.data();
        vk_check(vkAllocateDescriptorSets(device, &dsai, sets.data()), "vkAllocateDescriptorSets");

        VkDescriptorBufferInfo infos[5] {
            { a.buffer, 0, a.size },
            { b.buffer, 0, b.size },
            { d.buffer, 0, d.size },
            { fuse0.buffer, 0, fuse0.size },
            { fuse1.buffer, 0, fuse1.size },
        };
        for (uint32_t set_idx = 0; set_idx < chunk_count; ++set_idx) {
            VkWriteDescriptorSet writes[5] {};
            for (uint32_t i = 0; i < 5; ++i) {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = sets[set_idx];
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &infos[i];
            }
            vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
        }

        VkCommandPoolCreateInfo cpci2 { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        cpci2.queueFamilyIndex = queue_family;
        VkCommandPool cmd_pool;
        vk_check(vkCreateCommandPool(device, &cpci2, nullptr, &cmd_pool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cbai { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cbai.commandPool = cmd_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        vk_check(vkAllocateCommandBuffers(device, &cbai, &cmd), "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo cbi { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vk_check(vkBeginCommandBuffer(cmd, &cbi), "vkBeginCommandBuffer");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        for (uint32_t set_idx = 0; set_idx < chunk_count; ++set_idx) {
            const uint32_t row0 = set_idx * chunk_rows;
            const uint32_t chunk = std::min<uint32_t>(chunk_rows, rows - row0);
            uint32_t pc[13] {
                cols, row0, cols, rows,
                rows * cols, cols, rows,
                0, 0, 1, 1, 1, 1,
            };
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &sets[set_idx], 0, nullptr);
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
            vkCmdDispatch(cmd, chunk, 1, 1);
        }
        vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        VkSubmitInfo submit { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        vk_check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
        vk_check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");

        double max_err = 0.0;
        uint32_t max_row = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            double expect = 0.0;
            for (uint32_t c = 0; c < cols; ++c) {
                expect += double(pa[uint64_t(r) * cols + c]) * double(pb[c]);
            }
            const double err = std::fabs(double(pd[r]) - expect);
            if (err > max_err) {
                max_err = err;
                max_row = r;
            }
        }

        std::cout << "rows=" << rows << " cols=" << cols
                  << " max_err=" << max_err << " max_row=" << max_row << "\n";
        return max_err < 5e-4 ? 0 : 2;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
