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


#pragma once
#include "hdr_sampling.hpp"
#include "nvvk/gizmos_vk.hpp"
#include "renderer.h"

/*

 Structure of the application

    +--------------------------------------------+
    |             SampleExample                  |
    +--------+-----------------------------------+
    |  Pick  |    RtxPipeline   | other   ? ...  |
    +--------+---------+-------------------------+
    |       TLAS       |                         |
    +------------------+     Offscreen           |
    |      Scene       |                         |
    +------------------+-------------------------+

*/


// #define ALLOC_DMA  <--- This is in the CMakeLists.txt
#include "nvvk/resourceallocator_vk.hpp"
#if defined(ALLOC_DMA)
#include <nvvk/memallocator_dma_vk.hpp>
typedef nvvk::ResourceAllocatorDma Allocator;
#elif defined(ALLOC_VMA)
#include <nvvk/memallocator_vma_vk.hpp>
typedef nvvk::ResourceAllocatorVma Allocator;
#else
typedef nvvk::ResourceAllocatorDedicated Allocator;
#endif

#define CPP  // For sun_and_sky

#include "nvh/gltfscene.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/profiler_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/raypicker_vk.hpp"

#include "headless_vk.hpp"
#include "accelstruct.hpp"
#include "render_output.hpp"
#include "scene.hpp"
#include "shaders/host_device.h"

#include "queue.hpp"

class SampleExample : public HeadlessAppVK
{
public:
  enum RndMethod
  {
    eRtxPipeline,
    eRayQuery,
    eNone,
  };

  enum Queues
  {
    eGCT0,
    eGCT1,
    eCompute,
    eTransfer
  };

  void setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, const std::vector<nvvk::Queue>& queues, uint32_t width, uint32_t height, VkFormat colorFormat = VK_FORMAT_B8G8R8A8_UNORM, VkFormat depthFormat = VK_FORMAT_UNDEFINED);

  void createDescriptorSetLayout();
  void createUniformBuffer();
  void destroyResources();
  void loadAssets(const char* filename);
  void loadEnvironmentHdr(const std::string& hdrFilename);
  void loadScene(const std::string& filename);
  void createRender(RndMethod method);
  void resetFrame();
  void updateHdrDescriptors();
  void updateUniformBuffer(const VkCommandBuffer& cmdBuf);
  void dumpImage();

  Scene              m_scene;
  AccelStructure     m_accelStruct;
  RenderOutput       m_offscreen;
  HdrSampling        m_skydome;
  nvvk::RayPickerKHR m_picker;

  // It is possible that ray query isn't supported (ex. Titan)
  void supportRayQuery(bool support) { m_supportRayQuery = support; }
  bool m_supportRayQuery{true};

  // All renderers
  std::array<Renderer*, eNone> m_pRender{nullptr, nullptr};
  RndMethod                    m_rndMethod{eNone};

  nvvk::Buffer m_sunAndSkyBuffer;

  // Graphic pipeline
  VkDescriptorPool            m_descPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout       m_descSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet             m_descSet{VK_NULL_HANDLE};
  nvvk::DescriptorSetBindings m_bind;

  Allocator       m_alloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil m_debug;  // Utility to name objects


  VkRect2D m_renderRegion{};
  void     setRenderRegion(const VkRect2D& size);

  // #Post
  void createOffscreenRender();
  void drawPost(VkCommandBuffer cmdBuf);

  // #VKRay
  void renderScene(const VkCommandBuffer& cmdBuf, nvvk::ProfilerVK& profiler);


  RtxState m_rtxState{
      0,       // frame;
      10,      // maxDepth;
      1,       // maxSamples;
      1,       // fireflyClampThreshold;
      1,       // hdrMultiplier;
      0,       // debugging_mode;
      0,       // pbrMode;
      0,       // _pad0;
      {0, 0},  // size;
      0,       // minHeatmap;
      65000    // maxHeatmap;
  };

  SunAndSky m_sunAndSky{
      {1, 1, 1},            // rgb_unit_conversion;
      0.0000101320f,        // multiplier;
      0.0f,                 // haze;
      0.0f,                 // redblueshift;
      1.0f,                 // saturation;
      0.0f,                 // horizon_height;
      {0.4f, 0.4f, 0.4f},   // ground_color;
      0.1f,                 // horizon_blur;
      {0.0, 0.0, 0.01f},    // night_color;
      0.8f,                 // sun_disk_intensity;
      {0.00, 0.78, 0.62f},  // sun_direction;
      5.0f,                 // sun_disk_scale;
      1.0f,                 // sun_glow_intensity;
      1,                    // y_is_up;
      1,                    // physically_scaled_sun;
      0,                    // in_use;
  };

  int         m_maxFrames{100000};
  bool        m_showAxis{true};
  bool        m_descaling{false};
  int         m_descalingLevel{1};
};
