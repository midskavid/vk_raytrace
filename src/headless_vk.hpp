#pragma once

#include "vulkan/vulkan_core.h"
#include "nvmath/nvmath.h"
#include "nvh/nvprint.hpp"
#include "nvp/nvpsystem.hpp"

#ifdef LINUX
#include <unistd.h>
#endif

// GLFW
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#include "GLFW/glfw3.h"
#include "GLFW/glfw3native.h"

#include <cmath>
#include <set>
#include <vector>
#include <array>

struct HeadlessAppVKCreateInfo
{
  VkInstance            instance{};
  VkDevice              device{};
  VkPhysicalDevice      physicalDevice{};
  std::vector<uint32_t> queueIndices{};
  VkExtent2D            size{};
  bool                  useDynamicRendering{false};  // VK_KHR_dynamic_rendering
};


class HeadlessAppVK
{
public:
  HeadlessAppVK()          = default;
  virtual ~HeadlessAppVK() = default;

  virtual void setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t graphicsQueueIndex, uint32_t width, uint32_t height, VkFormat colorFormat = VK_FORMAT_B8G8R8A8_UNORM, VkFormat depthFormat = VK_FORMAT_UNDEFINED);

  virtual void destroy();
  virtual void createFrameBuffer();
  virtual void createRenderPass();
  virtual void createDepthBuffer();
  virtual void createColorBuffer();
  virtual void createCommandBuffer();
  virtual void submitWork(const VkCommandBuffer& cmdBuffer);  

  void         setViewport(const VkCommandBuffer& cmdBuf);
  void         fitCamera(const nvmath::vec3f& boxMin, const nvmath::vec3f& boxMax, bool instantFit = true);

  // Getters
  VkInstance                          getInstance() { return m_instance; }
  VkDevice                            getDevice() { return m_device; }
  VkPhysicalDevice                    getPhysicalDevice() { return m_physicalDevice; }
  VkQueue                             getQueue() { return m_queue; }
  uint32_t                            getQueueFamily() { return m_graphicsQueueIndex; }
  VkCommandPool                       getCommandPool() { return m_cmdPool; }
  VkRenderPass                        getRenderPass() { return m_renderPass; }
  VkExtent2D                          getSize() { return m_size; }
  VkPipelineCache                     getPipelineCache() { return m_pipelineCache; }
  VkFramebuffer                       getFramebuffer() { return m_framebuffer; }
  VkCommandBuffer                     getCommandBuffer() { return m_commandBuffer; }
  VkFormat                            getColorFormat() const { return m_colorFormat; }
  VkFormat                            getDepthFormat() const { return m_depthFormat; }
  VkImageView                         getDepthView() { return m_depthView; }

protected:
  uint32_t getMemoryType(uint32_t typeBits, const VkMemoryPropertyFlags& properties) const;

  VkCommandBuffer createTempCmdBuffer();
  void            submitTempCmdBuffer(VkCommandBuffer cmdBuffer);

  // Vulkan low level
  VkInstance       m_instance{};
  VkDevice         m_device{};
  VkPhysicalDevice m_physicalDevice{};
  VkQueue          m_queue{VK_NULL_HANDLE};
  uint32_t         m_graphicsQueueIndex{VK_QUEUE_FAMILY_IGNORED};
  VkCommandPool    m_cmdPool{VK_NULL_HANDLE};
  VkDescriptorPool m_imguiDescPool{VK_NULL_HANDLE};

  // Drawing/Surface
  VkFramebuffer                m_framebuffer;                    // Frame buffer
  VkCommandBuffer              m_commandBuffer;                  // Command buffer
  VkFence                      m_waitFence;                      // Fence
  VkImage                      m_colorImage{VK_NULL_HANDLE};     // Color
  VkDeviceMemory               m_colorMemory{VK_NULL_HANDLE};    // Color
  VkImageView                  m_colorView{VK_NULL_HANDLE};      // Color
  VkImage                      m_depthImage{VK_NULL_HANDLE};     // Depth/Stencil
  VkDeviceMemory               m_depthMemory{VK_NULL_HANDLE};    // Depth/Stencil
  VkImageView                  m_depthView{VK_NULL_HANDLE};      // Depth/Stencil
  VkRenderPass                 m_renderPass{VK_NULL_HANDLE};     // Base render pass
  VkExtent2D                   m_size{0, 0};                     // Size of the window
  VkPipelineCache              m_pipelineCache{VK_NULL_HANDLE};  // Cache for pipeline/shaders

  // Surface buffer formats
  VkFormat m_colorFormat{VK_FORMAT_B8G8R8A8_UNORM};
  VkFormat m_depthFormat{VK_FORMAT_UNDEFINED};

  bool  m_useDynamicRendering{false};  // Using VK_KHR_dynamic_rendering
  float m_sceneRadius{1.f};
};