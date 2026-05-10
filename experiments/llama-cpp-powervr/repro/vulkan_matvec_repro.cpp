#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <string>
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

static uint16_t fp32_to_fp16(float value) {
    union {
        float f;
        uint32_t u;
    } in { value };

    const uint32_t sign = (in.u >> 16) & 0x8000u;
    int32_t exp = int32_t((in.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = in.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) {
            return uint16_t(sign);
        }
        mant |= 0x800000u;
        const uint32_t shifted = mant >> uint32_t(14 - exp);
        return uint16_t(sign | ((shifted + 1u) >> 1));
    }
    if (exp >= 31) {
        return uint16_t(sign | 0x7c00u);
    }

    return uint16_t(sign | (uint32_t(exp) << 10) | ((mant + 0x1000u) >> 13));
}

static float fp16_to_fp32(uint16_t value) {
    const uint32_t sign = uint32_t(value & 0x8000u) << 16;
    uint32_t exp = (value >> 10) & 0x1fu;
    uint32_t mant = value & 0x03ffu;

    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    union {
        uint32_t u;
        float f;
    } result { out };
    return result.f;
}

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

static Buffer make_buffer(VkPhysicalDevice physical, VkDevice device, VkDeviceSize size,
                          VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VkMemoryPropertyFlags memory_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
    Buffer out;
    out.size = size;
    VkBufferCreateInfo bi { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(device, &bi, nullptr, &out.buffer), "vkCreateBuffer");

    VkMemoryRequirements req {};
    vkGetBufferMemoryRequirements(device, out.buffer, &req);
    VkMemoryAllocateInfo ai { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(physical, req.memoryTypeBits, memory_flags);
    vk_check(vkAllocateMemory(device, &ai, nullptr, &out.memory), "vkAllocateMemory");
    vk_check(vkBindBufferMemory(device, out.buffer, out.memory, 0), "vkBindBufferMemory");
    if (memory_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        vk_check(vkMapMemory(device, out.memory, 0, size, 0, &out.mapped), "vkMapMemory");
    }
    return out;
}

int main(int argc, char ** argv) {
    try {
        const uint32_t rows = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 1024;
        const uint32_t cols = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 1024;
        const char * spv_path = argc > 3 ? argv[3] : "matvec_scalar.spv";
        const std::string mode = argc > 4 ? argv[4] : "f32";
        const bool a_is_f16 = mode == "f16a";

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

        const VkDeviceSize align = 256;
        auto align_up = [align](VkDeviceSize v) {
            return ((v + align - 1) / align) * align;
        };
        const VkDeviceSize a_offset = align;
        const VkDeviceSize a_size = VkDeviceSize(rows) * cols * (a_is_f16 ? sizeof(uint16_t) : sizeof(float));
        const VkDeviceSize b_offset = align_up(a_offset + a_size);
        const VkDeviceSize b_size = VkDeviceSize(cols) * sizeof(float);
        const VkDeviceSize d_offset = align_up(b_offset + b_size);
        const VkDeviceSize d_size = VkDeviceSize(rows) * sizeof(float);
        const VkDeviceSize fuse0_offset = align_up(d_offset + d_size);
        const VkDeviceSize fuse1_offset = align_up(fuse0_offset + sizeof(float));
        const VkDeviceSize arena_size = align_up(fuse1_offset + sizeof(float));
        Buffer staging = make_buffer(physical, device, arena_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        Buffer arena = make_buffer(physical, device, arena_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer readback = make_buffer(physical, device, d_size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        auto * base = static_cast<uint8_t *>(staging.mapped);
        float * pa_f32 = reinterpret_cast<float *>(base + a_offset);
        uint16_t * pa_f16 = reinterpret_cast<uint16_t *>(base + a_offset);
        float * pb = reinterpret_cast<float *>(base + b_offset);
        float * pd = reinterpret_cast<float *>(base + d_offset);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                const int v = int((r * 17 + c * 13) % 2001) - 1000;
                const float a_value = float(v) / 1000.0f;
                if (a_is_f16) {
                    pa_f16[uint64_t(r) * cols + c] = fp32_to_fp16(a_value);
                } else {
                    pa_f32[uint64_t(r) * cols + c] = a_value;
                }
            }
        }
        for (uint32_t c = 0; c < cols; ++c) {
            const int v = int((c * 19) % 2001) - 1000;
            pb[c] = float(v) / 1000.0f;
        }
        for (uint32_t r = 0; r < rows; ++r) {
            pd[r] = 0.0f;
        }
        *reinterpret_cast<float *>(base + fuse0_offset) = 0.0f;
        *reinterpret_cast<float *>(base + fuse1_offset) = 0.0f;

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
            { arena.buffer, a_offset, a_size },
            { arena.buffer, b_offset, b_size },
            { arena.buffer, d_offset, d_size },
            { arena.buffer, fuse0_offset, sizeof(float) },
            { arena.buffer, fuse1_offset, sizeof(float) },
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
        VkBufferCopy upload_region {};
        upload_region.srcOffset = 0;
        upload_region.dstOffset = 0;
        upload_region.size = arena_size;
        vkCmdCopyBuffer(cmd, staging.buffer, arena.buffer, 1, &upload_region);

        VkMemoryBarrier upload_barrier { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        upload_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        upload_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &upload_barrier, 0, nullptr, 0, nullptr);

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

        VkMemoryBarrier shader_barrier { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        shader_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        shader_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &shader_barrier, 0, nullptr, 0, nullptr);

        VkBufferCopy read_region {};
        read_region.srcOffset = d_offset;
        read_region.dstOffset = 0;
        read_region.size = d_size;
        vkCmdCopyBuffer(cmd, arena.buffer, readback.buffer, 1, &read_region);

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
                const float a_value = a_is_f16 ? fp16_to_fp32(pa_f16[uint64_t(r) * cols + c]) : pa_f32[uint64_t(r) * cols + c];
                expect += double(a_value) * double(pb[c]);
            }
            const float got = static_cast<float *>(readback.mapped)[r];
            const double err = std::fabs(double(got) - expect);
            if (err > max_err) {
                max_err = err;
                max_row = r;
            }
        }

        std::cout << "mode=" << mode << " rows=" << rows << " cols=" << cols
                  << " max_err=" << max_err << " max_row=" << max_row << "\n";
        return max_err < (a_is_f16 ? 2e-2 : 5e-4) ? 0 : 2;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
