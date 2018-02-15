/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

namespace marian {
 
// TODO: A better approach to dispatch
enum ResidentDevice { DEVICE_CPU, DEVICE_GPU };

template <typename T> struct residency_trait;

}
