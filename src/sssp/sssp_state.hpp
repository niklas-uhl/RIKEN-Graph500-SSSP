/*
 * sssp_state.hpp
 *
 *  Created on: May 5, 2022
 *      Author: Daniel Rehfeldt
 */

#ifndef SRC_SSSP_SSSP_STATE_HPP_
#define SRC_SSSP_SSSP_STATE_HPP_

#include "parameters.h"
#include "utils_core.hpp"

// To transfer information about the state of an SSSP object
struct SsspState {
   const BitmapType* const vertices_is_settled_;
   float bucket_upper;
   bool is_bellman_ford_;
   bool is_light_phase_;
   bool with_settled_;
   bool is_presolving_mode_;

   static inline
   bool target_is_settled(const BitmapType* vertices_is_settled, int64_t tgt, int r_bits, int lgl, int64_t L) {
      const TwodVertex bit_idx = SeparatedId(SeparatedId(tgt).low(r_bits + lgl)).compact(lgl, L);
      return ( vertices_is_settled[bit_idx >> PRM::LOG_NBPE] & (BitmapType(1) << (bit_idx & PRM::NBPE_MASK)) );
   }
};



#endif /* SRC_SSSP_SSSP_STATE_HPP_ */
