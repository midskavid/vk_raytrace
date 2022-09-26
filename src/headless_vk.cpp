#include "headless_vk.hpp"
#include "nvp/perproject_globals.hpp"

void HeadlessAppVK::setup(const VkInstance& instance, 
                       const VkDevice& device,
                       const VkPhysicalDevice& physicalDevice,
                       uint32_t graphicsQueueIndex,
                       uint32_t width,
                       uint32_t height,
                       VkFormat colorFormat /*= VK_FORMAT_B8G8R8A8_UNORM*/,
                       VkFormat depthFormat /*= VK_FORMAT_UNDEFINED*/)

{
  m_instance           = instance;
  m_device             = device;
  m_physicalDevice     = physicalDevice;
  m_graphicsQueueIndex = graphicsQueueIndex;
  m_size               = VkExtent2D{width, height};
  m_colorFormat        = colorFormat;
  m_depthFormat        = depthFormat;

  vkGetDeviceQueue(m_device, m_graphicsQueueIndex, 0, &m_queue);

  VkCommandPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  vkCreateCommandPool(m_device, &poolCreateInfo, nullptr, &m_cmdPool);

  VkPipelineCacheCreateInfo pipelineCacheInfo{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  vkCreatePipelineCache(m_device, &pipelineCacheInfo, nullptr, &m_pipelineCache);

  // Find the most suitable depth format
  if(m_depthFormat == VK_FORMAT_UNDEFINED)
  {
    auto feature = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
    for(const auto& f : {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT})
    {
      VkFormatProperties formatProp{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
      vkGetPhysicalDeviceFormatProperties(m_physicalDevice, f, &formatProp);
      if((formatProp.optimalTilingFeatures & feature) == feature)
      {
        m_depthFormat = f;
        break;
      }
    }
  }

}

//--------------------------------------------------------------------------------------------------
// To call on exit
//
void HeadlessAppVK::destroy()
{
  vkDeviceWaitIdle(m_device);

  if(!m_useDynamicRendering)
    vkDestroyRenderPass(m_device, m_renderPass, nullptr);

  vkDestroyImageView(m_device, m_depthView, nullptr);
  vkDestroyImage(m_device, m_depthImage, nullptr);
  vkFreeMemory(m_device, m_depthMemory, nullptr);
  vkDestroyPipelineCache(m_device, m_pipelineCache, nullptr);

  vkDestroyImageView(m_device, m_colorView, nullptr);
  vkDestroyImage(m_device, m_colorImage, nullptr);
  vkFreeMemory(m_device, m_colorMemory, nullptr);

  vkDestroyFramebuffer(m_device, m_framebuffer, nullptr);
  vkFreeCommandBuffers(m_device, m_cmdPool, 1, &m_commandBuffer);
  vkDestroyFence(m_device, m_waitFence, nullptr);

  vkDestroyDescriptorPool(m_device, m_imguiDescPool, nullptr);
  vkDestroyCommandPool(m_device, m_cmdPool, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Create all the framebuffers in which the image will be rendered
// - Swapchain need to be created before calling this
//
void HeadlessAppVK::createFrameBuffer()
{
  if(m_useDynamicRendering)
    return;

  // Recreate the frame buffer
  vkDestroyFramebuffer(m_device, m_framebuffer, nullptr);

  // Array of attachment (color, depth)
  std::array<VkImageView, 2> attachments{};

  VkFramebufferCreateInfo framebufferCreateInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  framebufferCreateInfo.renderPass      = m_renderPass;
  framebufferCreateInfo.attachmentCount = 2;
  framebufferCreateInfo.width           = m_size.width;
  framebufferCreateInfo.height          = m_size.height;
  framebufferCreateInfo.layers          = 1;
  framebufferCreateInfo.pAttachments    = attachments.data();
  
  attachments[0] = m_colorView;
  attachments[1] = m_depthView;
  vkCreateFramebuffer(m_device, &framebufferCreateInfo, nullptr, &m_framebuffer);
}

void HeadlessAppVK::createRenderPass()
{
  if(m_useDynamicRendering)
    return;

  if(m_renderPass)
    vkDestroyRenderPass(m_device, m_renderPass, nullptr);

  std::array<VkAttachmentDescription, 2> attachments{};
  // Color attachment
  attachments[0].format      = m_colorFormat;
  attachments[0].loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[0].storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  attachments[0].samples     = VK_SAMPLE_COUNT_1_BIT;

  // Depth attachment
  attachments[1].format        = m_depthFormat;
  attachments[1].loadOp        = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachments[1].finalLayout   = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  attachments[1].samples       = VK_SAMPLE_COUNT_1_BIT;

  // One color, one depth
  const VkAttachmentReference colorReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  const VkAttachmentReference depthReference{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

  std::array<VkSubpassDependency, 2> subpassDependencies{};
  // Transition from final to initial (VK_SUBPASS_EXTERNAL refers to all commands executed outside of the actual renderpass)
  subpassDependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
  subpassDependencies[0].dstSubpass      = 0;
  subpassDependencies[0].srcStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  subpassDependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  subpassDependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
  subpassDependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  subpassDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

  subpassDependencies[1].srcSubpass = 0;
  subpassDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  subpassDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  subpassDependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  subpassDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  subpassDependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
  subpassDependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

  VkSubpassDescription subpassDescription{};
  subpassDescription.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpassDescription.colorAttachmentCount    = 1;
  subpassDescription.pColorAttachments       = &colorReference;
  subpassDescription.pDepthStencilAttachment = &depthReference;

  VkRenderPassCreateInfo renderPassInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
  renderPassInfo.pAttachments    = attachments.data();
  renderPassInfo.subpassCount    = 1;
  renderPassInfo.pSubpasses      = &subpassDescription;
  renderPassInfo.dependencyCount = static_cast<uint32_t>(subpassDependencies.size());
  renderPassInfo.pDependencies   = subpassDependencies.data();

  vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass);

#ifdef _DEBUG
  VkDebugUtilsObjectNameInfoEXT nameInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
  nameInfo.objectHandle = (uint64_t)m_renderPass;
  nameInfo.objectType   = VK_OBJECT_TYPE_RENDER_PASS;
  nameInfo.pObjectName  = R"(AppBaseVk)";
  vkSetDebugUtilsObjectNameEXT(m_device, &nameInfo);
#endif  // _DEBUG
}

//--------------------------------------------------------------------------------------------------
// Create an image to be used as color buffer
//
void HeadlessAppVK::createColorBuffer()
{
  if(m_colorView)
    vkDestroyImageView(m_device, m_colorView, nullptr);

  if(m_colorImage)
    vkDestroyImage(m_device, m_colorImage, nullptr);

  if(m_colorMemory)
    vkFreeMemory(m_device, m_colorMemory, nullptr);


  VkImageCreateInfo        image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  image.imageType   = VK_IMAGE_TYPE_2D;
  image.extent      = VkExtent3D{m_size.width, m_size.height, 1};
  image.format      = m_colorFormat;
  image.mipLevels   = 1;
  image.arrayLayers = 1;
  image.samples     = VK_SAMPLE_COUNT_1_BIT;
  image.tiling      = VK_IMAGE_TILING_OPTIMAL;
  image.usage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  // Create the color image
  
  vkCreateImage(m_device, &image, nullptr, &m_colorImage);

  // Allocate the memory
  VkMemoryRequirements memReqs;
  vkGetImageMemoryRequirements(m_device, m_colorImage, &memReqs);
  VkMemoryAllocateInfo memAllocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  memAllocInfo.allocationSize  = memReqs.size;
  memAllocInfo.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkAllocateMemory(m_device, &memAllocInfo, nullptr, &m_colorMemory);

  // Bind image and memory
  vkBindImageMemory(m_device, m_colorImage, m_colorMemory, 0);

  // Setting up the view
  VkImageViewCreateInfo colorImageView{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  colorImageView.viewType                         = VK_IMAGE_VIEW_TYPE_2D;
  colorImageView.format                           = m_colorFormat;
  colorImageView.subresourceRange                 = {};
  colorImageView.subresourceRange.aspectMask      = VK_IMAGE_ASPECT_COLOR_BIT;
  colorImageView.subresourceRange.baseMipLevel    = 0;
  colorImageView.subresourceRange.levelCount      = 1;
  colorImageView.subresourceRange.baseArrayLayer  = 0;
  colorImageView.subresourceRange.layerCount      = 1;
  colorImageView.image                            = m_colorImage;
  vkCreateImageView(m_device, &colorImageView, nullptr, &m_colorView);  
}

//--------------------------------------------------------------------------------------------------
// Creating an image to be used as depth buffer
//
void HeadlessAppVK::createDepthBuffer()
{
  if(m_depthView)
    vkDestroyImageView(m_device, m_depthView, nullptr);

  if(m_depthImage)
    vkDestroyImage(m_device, m_depthImage, nullptr);

  if(m_depthMemory)
    vkFreeMemory(m_device, m_depthMemory, nullptr);

  // Depth information
  const VkImageAspectFlags aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  VkImageCreateInfo        depthStencilCreateInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  depthStencilCreateInfo.imageType   = VK_IMAGE_TYPE_2D;
  depthStencilCreateInfo.extent      = VkExtent3D{m_size.width, m_size.height, 1};
  depthStencilCreateInfo.format      = m_depthFormat;
  depthStencilCreateInfo.mipLevels   = 1;
  depthStencilCreateInfo.arrayLayers = 1;
  depthStencilCreateInfo.samples     = VK_SAMPLE_COUNT_1_BIT;
  depthStencilCreateInfo.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  // Create the depth image
  vkCreateImage(m_device, &depthStencilCreateInfo, nullptr, &m_depthImage);

#ifdef _DEBUG
  std::string                   name = std::string("AppBaseDepth");
  VkDebugUtilsObjectNameInfoEXT nameInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
  nameInfo.objectHandle = (uint64_t)m_depthImage;
  nameInfo.objectType   = VK_OBJECT_TYPE_IMAGE;
  nameInfo.pObjectName  = R"(AppBase)";
  vkSetDebugUtilsObjectNameEXT(m_device, &nameInfo);
#endif  // _DEBUG

  // Allocate the memory
  VkMemoryRequirements memReqs;
  vkGetImageMemoryRequirements(m_device, m_depthImage, &memReqs);
  VkMemoryAllocateInfo memAllocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  memAllocInfo.allocationSize  = memReqs.size;
  memAllocInfo.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkAllocateMemory(m_device, &memAllocInfo, nullptr, &m_depthMemory);

  // Bind image and memory
  vkBindImageMemory(m_device, m_depthImage, m_depthMemory, 0);

  // Setting up the view
  VkImageViewCreateInfo depthStencilView{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  depthStencilView.viewType         = VK_IMAGE_VIEW_TYPE_2D;
  depthStencilView.format           = m_depthFormat;
  depthStencilView.flags            = 0;
  depthStencilView.subresourceRange = {};
  
  depthStencilView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  if (m_depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT)
    depthStencilView.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
  
  depthStencilView.subresourceRange.baseMipLevel = 0;
  depthStencilView.subresourceRange.levelCount = 1;
  depthStencilView.subresourceRange.baseArrayLayer = 0;
  depthStencilView.subresourceRange.layerCount = 1;

  depthStencilView.image            = m_depthImage;
  vkCreateImageView(m_device, &depthStencilView, nullptr, &m_depthView);
}

void HeadlessAppVK::createCommandBuffer() 
{
  if (m_commandBuffer)
    vkFreeCommandBuffers(m_device, m_cmdPool, 1, &m_commandBuffer);

  VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocateInfo.commandBufferCount = 1;
  allocateInfo.commandPool        = m_cmdPool;
  allocateInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  
  vkAllocateCommandBuffers(m_device, &allocateInfo, &m_commandBuffer);
  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(m_commandBuffer, &beginInfo);
}

void HeadlessAppVK::submitWork(const VkCommandBuffer& cmdBuffer)
{
  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmdBuffer;
  VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  vkCreateFence(m_device, &fenceInfo, nullptr, &fence);
  vkQueueSubmit(m_queue, 1, &submitInfo, fence);
  vkWaitForFences(m_device, 1, &fence, VK_TRUE, UINT64_MAX);
  vkDestroyFence(m_device, fence, nullptr);
}

//--------------------------------------------------------------------------------------------------
// When the pipeline is set for using dynamic, this becomes useful
//
void HeadlessAppVK::setViewport(const VkCommandBuffer& cmdBuf)
{
  VkViewport viewport{0.0f, 0.0f, static_cast<float>(m_size.width), static_cast<float>(m_size.height), 0.0f, 1.0f};
  vkCmdSetViewport(cmdBuf, 0, 1, &viewport);

  VkRect2D scissor{{0, 0}, {m_size.width, m_size.height}};
  vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
}

uint32_t HeadlessAppVK::getMemoryType(uint32_t typeBits, const VkMemoryPropertyFlags& properties) const
{
  VkPhysicalDeviceMemoryProperties memoryProperties;
  vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memoryProperties);

  for(uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
  {
    if(((typeBits & (1 << i)) > 0) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
      return i;
  }
  std::string err = "Unable to find memory type " + std::to_string(properties);
  LOGE(err.c_str());
  assert(0);
  return ~0u;
}

VkCommandBuffer HeadlessAppVK::createTempCmdBuffer()
{
  // Create an image barrier to change the layout from undefined to DepthStencilAttachmentOptimal
  VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocateInfo.commandBufferCount = 1;
  allocateInfo.commandPool        = m_cmdPool;
  allocateInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  VkCommandBuffer cmdBuffer;
  vkAllocateCommandBuffers(m_device, &allocateInfo, &cmdBuffer);

  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  //beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmdBuffer, &beginInfo);
  return cmdBuffer;
}

void HeadlessAppVK::submitTempCmdBuffer(VkCommandBuffer cmdBuffer)
{
  vkEndCommandBuffer(cmdBuffer);

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers    = &cmdBuffer;
  vkQueueSubmit(m_queue, 1, &submitInfo, {});
  vkQueueWaitIdle(m_queue);
  vkFreeCommandBuffers(m_device, m_cmdPool, 1, &cmdBuffer);
}
