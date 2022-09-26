/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


/*
 * Main class to render the scene, holds sub-classes for various work
 */


#define VMA_IMPLEMENTATION

#include <fstream>
#include <string>

#include "shaders/host_device.h"
#include "rayquery.hpp"
#include "rtx_pipeline.hpp"
#include "sample_example.hpp"
#include "tools.hpp"

#include "nvml_monitor.hpp"
#include "fileformats/tiny_gltf_freeimage.h"


#if defined(NVP_SUPPORTS_NVML)
NvmlMonitor g_nvml(100, 100);
#endif

//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void SampleExample::setup(const VkInstance&               instance,
                          const VkDevice&                 device,
                          const VkPhysicalDevice&         physicalDevice,
                          const std::vector<nvvk::Queue>& queues,
                          uint32_t width,
                          uint32_t height,
                          VkFormat colorFormat /*= VK_FORMAT_B8G8R8A8_UNORM*/,
                          VkFormat depthFormat /*= VK_FORMAT_UNDEFINED*/)

{
  HeadlessAppVK::setup(instance, device, physicalDevice, queues[eGCT0].familyIndex, width, height, colorFormat, depthFormat);

  // Memory allocator for buffers and images
  m_alloc.init(instance, device, physicalDevice);

  m_debug.setup(m_device);

  // Compute queues can be use for acceleration structures
  m_picker.setup(m_device, physicalDevice, queues[eCompute].familyIndex, &m_alloc);
  m_accelStruct.setup(m_device, physicalDevice, queues[eCompute].familyIndex, &m_alloc);

  // Note: the GTC family queue is used because the nvvk::cmdGenerateMipmaps uses vkCmdBlitImage and this
  // command requires graphic queue and not only transfer.
  m_scene.setup(m_device, physicalDevice, queues[eGCT1], &m_alloc);

  // Transfer queues can be use for the creation of the following assets
  m_offscreen.setup(m_device, physicalDevice, queues[eTransfer].familyIndex, &m_alloc);
  m_skydome.setup(device, physicalDevice, queues[eTransfer].familyIndex, &m_alloc);

  // Create and setup all renderers
  m_pRender[eRtxPipeline] = new RtxPipeline;
  m_pRender[eRayQuery]    = new RayQuery;
  for(auto r : m_pRender)
  {
    r->setup(m_device, physicalDevice, queues[eTransfer].familyIndex, &m_alloc);
  }
}


//--------------------------------------------------------------------------------------------------
// Loading the scene file, setting up all scene buffers, create the acceleration structures
// for the loaded models.
//
void SampleExample::loadScene(const std::string& filename)
{
  m_scene.load(filename);
  m_accelStruct.create(m_scene.getScene(), m_scene.getBuffers(Scene::eVertex), m_scene.getBuffers(Scene::eIndex));

  // The picker is the helper to return information from a ray hit under the mouse cursor
  m_picker.setTlas(m_accelStruct.getTlas());
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Loading an HDR image and creating the importance sampling acceleration structure
//
void SampleExample::loadEnvironmentHdr(const std::string& hdrFilename)
{
  MilliTimer timer;
  LOGI("Loading HDR and converting %s\n", hdrFilename.c_str());
  m_skydome.loadEnvironment(hdrFilename);
  timer.print();

  m_rtxState.fireflyClampThreshold = m_skydome.getIntegral() * 4.f;  // magic
}


//--------------------------------------------------------------------------------------------------
// Called at each frame to update the UBO: scene, camera, environment (sun&sky)
//
void SampleExample::updateUniformBuffer(const VkCommandBuffer& cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);
  const float aspectRatio = m_renderRegion.extent.width / static_cast<float>(m_renderRegion.extent.height);

  m_scene.updateCamera(cmdBuf, aspectRatio);
  vkCmdUpdateBuffer(cmdBuf, m_sunAndSkyBuffer.buffer, 0, sizeof(SunAndSky), &m_sunAndSky);
}

//--------------------------------------------------------------------------------------------------
// Reset frame is re-starting the rendering
//
void SampleExample::resetFrame()
{
  m_rtxState.frame = -1;
}

//--------------------------------------------------------------------------------------------------
// Descriptors for the Sun&Sky buffer
//
void SampleExample::createDescriptorSetLayout()
{
  VkShaderStageFlags flags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
                             | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;


  m_bind.addBinding({EnvBindings::eSunSky, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_MISS_BIT_KHR | flags});
  m_bind.addBinding({EnvBindings::eHdr, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, flags});  // HDR image
  m_bind.addBinding({EnvBindings::eImpSamples, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flags});   // importance sampling


  m_descPool = m_bind.createPool(m_device, 1);
  CREATE_NAMED_VK(m_descSetLayout, m_bind.createLayout(m_device));
  CREATE_NAMED_VK(m_descSet, nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));

  // Using the environment
  std::vector<VkWriteDescriptorSet> writes;
  VkDescriptorBufferInfo            sunskyDesc{m_sunAndSkyBuffer.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo            accelImpSmpl{m_skydome.m_accelImpSmpl.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_bind.makeWrite(m_descSet, EnvBindings::eSunSky, &sunskyDesc));
  writes.emplace_back(m_bind.makeWrite(m_descSet, EnvBindings::eHdr, &m_skydome.m_texHdr.descriptor));
  writes.emplace_back(m_bind.makeWrite(m_descSet, EnvBindings::eImpSamples, &accelImpSmpl));

  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Setting the descriptor for the HDR and its acceleration structure
//
void SampleExample::updateHdrDescriptors()
{
  std::vector<VkWriteDescriptorSet> writes;
  VkDescriptorBufferInfo            accelImpSmpl{m_skydome.m_accelImpSmpl.buffer, 0, VK_WHOLE_SIZE};

  writes.emplace_back(m_bind.makeWrite(m_descSet, EnvBindings::eHdr, &m_skydome.m_texHdr.descriptor));
  writes.emplace_back(m_bind.makeWrite(m_descSet, EnvBindings::eImpSamples, &accelImpSmpl));
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the Sun&Sky structure
// - Buffer is host visible and will be set each frame
//
void SampleExample::createUniformBuffer()
{
  m_sunAndSkyBuffer = m_alloc.createBuffer(sizeof(SunAndSky), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  NAME_VK(m_sunAndSkyBuffer.buffer);
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void SampleExample::destroyResources()
{
  // Resources
  m_alloc.destroy(m_sunAndSkyBuffer);

  // Descriptors
  vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

  // Other
  m_picker.destroy();
  m_scene.destroy();
  m_accelStruct.destroy();
  m_offscreen.destroy();
  m_skydome.destroy();

  // All renderers
  for(auto p : m_pRender)
  {
    p->destroy();
    p = nullptr;
  }

  // Memory
  m_alloc.deinit();
}

//--------------------------------------------------------------------------------------------------
// Creating the render: RTX, Ray Query, ...
// - Destroy the previous one.
void SampleExample::createRender(RndMethod method)
{
  if(method == m_rndMethod)
    return;

  LOGI("Switching renderer, from %d to %d \n", m_rndMethod, method);
  if(m_rndMethod != eNone)
  {
    vkDeviceWaitIdle(m_device);  // cannot destroy while in use
    m_pRender[m_rndMethod]->destroy();
  }
  m_rndMethod = method;

  m_pRender[m_rndMethod]->create(
      m_size, {m_accelStruct.getDescLayout(), m_offscreen.getDescLayout(), m_scene.getDescLayout(), m_descSetLayout}, &m_scene);
}

//--------------------------------------------------------------------------------------------------
// The GUI is taking space and size of the rendering area is smaller than the viewport
// This is the space left in the center view.
void SampleExample::setRenderRegion(const VkRect2D& size)
{
  if(memcmp(&m_renderRegion, &size, sizeof(VkRect2D)) != 0)
    resetFrame();
  m_renderRegion = size;
}

//////////////////////////////////////////////////////////////////////////
// Post ray tracing
//////////////////////////////////////////////////////////////////////////

void SampleExample::createOffscreenRender()
{
  m_offscreen.create(m_size, m_renderPass);
}

//--------------------------------------------------------------------------------------------------
// This will draw the result of the rendering and apply the tonemapper.
// If enabled, draw orientation axis in the lower left corner.
void SampleExample::drawPost(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);
  auto size = nvmath::vec2f(m_size.width, m_size.height);
  auto area = nvmath::vec2f(m_renderRegion.extent.width, m_renderRegion.extent.height);

  VkViewport viewport{static_cast<float>(m_renderRegion.offset.x),
                      static_cast<float>(m_renderRegion.offset.y),
                      static_cast<float>(m_size.width),
                      static_cast<float>(m_size.height),
                      0.0f,
                      1.0f};
  VkRect2D   scissor{};
  scissor.extent = {m_size.width, m_size.height};
  vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
  vkCmdSetScissor(cmdBuf, 0, 1, &scissor);

  m_offscreen.m_tonemapper.zoom           = m_descaling ? 1.0f / m_descalingLevel : 1.0f;
  m_offscreen.m_tonemapper.renderingRatio = 1.0f;
  m_offscreen.run(cmdBuf);

}

//////////////////////////////////////////////////////////////////////////
// Ray tracing
//////////////////////////////////////////////////////////////////////////

void SampleExample::renderScene(const VkCommandBuffer& cmdBuf, nvvk::ProfilerVK& profiler)
{
#if defined(NVP_SUPPORTS_NVML)
  g_nvml.refresh();
#endif

  LABEL_SCOPE_VK(cmdBuf);

  auto sec = profiler.timeRecurring("Render", cmdBuf);

  // We are done rendering
  if(m_rtxState.frame >= m_maxFrames)
    return;

  // Handling de-scaling by reducing the size to render
  VkExtent2D render_size = m_renderRegion.extent;
  if(m_descaling)
    render_size = VkExtent2D{render_size.width / m_descalingLevel, render_size.height / m_descalingLevel};

  m_rtxState.size = {m_size.width, m_size.height};
  // State is the push constant structure
  m_pRender[m_rndMethod]->setPushContants(m_rtxState);
  // Running the renderer
  m_pRender[m_rndMethod]->run(cmdBuf, render_size, profiler,
                              {m_accelStruct.getDescSet(), m_offscreen.getDescSet(), m_scene.getDescSet(), m_descSet});


  // For automatic brightness tonemapping
  if(m_offscreen.m_tonemapper.autoExposure)
  {
    auto slot = profiler.timeRecurring("Mipmap", cmdBuf);
    m_offscreen.genMipmap(cmdBuf);
  }
}

void insertImageMemoryBarrier(
  VkCommandBuffer cmdbuffer,
  VkImage image,
  VkAccessFlags srcAccessMask,
  VkAccessFlags dstAccessMask,
  VkImageLayout oldImageLayout,
  VkImageLayout newImageLayout,
  VkPipelineStageFlags srcStageMask,
  VkPipelineStageFlags dstStageMask,
  VkImageSubresourceRange subresourceRange)
{
  VkImageMemoryBarrier imageMemoryBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  imageMemoryBarrier.srcAccessMask = srcAccessMask;
  imageMemoryBarrier.dstAccessMask = dstAccessMask;
  imageMemoryBarrier.oldLayout = oldImageLayout;
  imageMemoryBarrier.newLayout = newImageLayout;
  imageMemoryBarrier.image = image;
  imageMemoryBarrier.subresourceRange = subresourceRange;

  vkCmdPipelineBarrier(
    cmdbuffer,
    srcStageMask,
    dstStageMask,
    0,
    0, nullptr,
    0, nullptr,
    1, &imageMemoryBarrier);
}

void SampleExample::dumpImage()
{
	/*
			Copy framebuffer image to host visible image
	*/
  const char* imagedata;
  {
    // Create the linear tiled destination image to copy to and to read the memory from
    VkImageCreateInfo imgCreateInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imgCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imgCreateInfo.extent.width = m_size.width;
    imgCreateInfo.extent.height = m_size.height;
    imgCreateInfo.extent.depth = 1;
    imgCreateInfo.arrayLayers = 1;
    imgCreateInfo.mipLevels = 1;
    imgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgCreateInfo.tiling = VK_IMAGE_TILING_LINEAR;
    imgCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    // Create the image
    VkImage dstImage;
    vkCreateImage(m_device, &imgCreateInfo, nullptr, &dstImage);
    // Create memory to back up the image
    VkMemoryRequirements memRequirements;
    VkMemoryAllocateInfo memAllocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    VkDeviceMemory dstImageMemory;
    vkGetImageMemoryRequirements(m_device, dstImage, &memRequirements);
    memAllocInfo.allocationSize = memRequirements.size;
    // Memory must be host visible to copy from
    memAllocInfo.memoryTypeIndex = getMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(m_device, &memAllocInfo, nullptr, &dstImageMemory);
    vkBindImageMemory(m_device, dstImage, dstImageMemory, 0);

    // Do the actual blit from the offscreen image to our host visible destination image
    VkCommandBufferAllocateInfo cmdBufAllocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdBufAllocateInfo.commandPool        = m_cmdPool;
    cmdBufAllocateInfo.commandBufferCount = 1;
    cmdBufAllocateInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    VkCommandBuffer copyCmd;
    vkAllocateCommandBuffers(m_device, &cmdBufAllocateInfo, &copyCmd);
    VkCommandBufferBeginInfo cmdBufInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};;
    vkBeginCommandBuffer(copyCmd, &cmdBufInfo);

    // Transition destination image to transfer destination layout
    insertImageMemoryBarrier(
    	copyCmd,
    	dstImage,
    	0,
    	VK_ACCESS_TRANSFER_WRITE_BIT,
    	VK_IMAGE_LAYOUT_UNDEFINED,
    	VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    	VK_PIPELINE_STAGE_TRANSFER_BIT,
    	VK_PIPELINE_STAGE_TRANSFER_BIT,
    	VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

    // colorAttachment.image is already in VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, and does not need to be transitioned

    VkImageCopy imageCopyRegion{};
    imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageCopyRegion.srcSubresource.layerCount = 1;
    imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageCopyRegion.dstSubresource.layerCount = 1;
    imageCopyRegion.extent.width = m_size.width;
    imageCopyRegion.extent.height = m_size.height;
    imageCopyRegion.extent.depth = 1;

    vkCmdCopyImage(
      copyCmd,
      m_colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      1,
      &imageCopyRegion);

    // // Transition destination image to general layout, which is the required layout for mapping the image memory later on
    insertImageMemoryBarrier(
    	copyCmd,
    	dstImage,
    	VK_ACCESS_TRANSFER_WRITE_BIT,
    	VK_ACCESS_MEMORY_READ_BIT,
    	VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    	VK_IMAGE_LAYOUT_GENERAL,
    	VK_PIPELINE_STAGE_TRANSFER_BIT,
    	VK_PIPELINE_STAGE_TRANSFER_BIT,
    	VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

    vkEndCommandBuffer(copyCmd);

    submitWork(copyCmd);

    // Get layout of the image (including row pitch)
    VkImageSubresource subResource{};
    subResource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    VkSubresourceLayout subResourceLayout;

    vkGetImageSubresourceLayout(m_device, dstImage, &subResource, &subResourceLayout);

    // Map image memory so we can start copying from it
    vkMapMemory(m_device, dstImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&imagedata);
    imagedata += subResourceLayout.offset;

    /*
      Save host visible framebuffer image to disk (ppm format)
    */


    const char* filename = "headless.ppm";

    std::ofstream file(filename, std::ios::out | std::ios::binary);

    // ppm header
    file << "P6\n" << m_size.width << "\n" << m_size.height << "\n" << 255 << "\n";

    std::vector<VkFormat> formatsBGR = { VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SNORM };
    bool colorSwizzle = (std::find(formatsBGR.begin(), formatsBGR.end(), VK_FORMAT_R8G8B8A8_UNORM) != formatsBGR.end());
    colorSwizzle = true;
    
    // ppm binary pixel data
    for (int32_t y = 0; y < m_size.height; y++) {
      unsigned int *row = (unsigned int*)imagedata;
      for (int32_t x = 0; x < m_size.width; x++) {
        if (colorSwizzle) {
          file.write((char*)row + 2, 1);
          file.write((char*)row + 1, 1);
          file.write((char*)row, 1);
        }
        else {
          file.write((char*)row, 3);
        }
        row++;
      }
      imagedata += subResourceLayout.rowPitch;
    }
    file.close();

    LOGI("Framebuffer image saved to %s\n", filename);

    // Clear buffers
    vkDestroyImage(m_device, dstImage, nullptr);
    vkFreeMemory(m_device, dstImageMemory, nullptr);
    vkFreeCommandBuffers(m_device, m_cmdPool, 1, &copyCmd);
  }
}