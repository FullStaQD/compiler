#include "fromEnzyme.h"

// NOLINTBEGIN
namespace qcc {

/// From Enzyme/MLIR/Interfaces/Utils.h (and Enzyme/MLIR/Interfaces/Utils.cpp)
bool isReadOnly(mlir::Operation* op) {
  if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance, 1> effects;
    effectInterface.getEffects(effects);
    if (!llvm::all_of(effects, [](const auto& it) { return isa<MemoryEffects::Read>(it.getEffect()); }))
      return false;
  }
  // For ops with HasRecursiveMemoryEffects, recurse into nested regions
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (auto& nestedOp : op->getRegions()...)
      if (!isReadOnly(&nestedOp))
        return false;
  }
  return true;
}

} // namespace qcc
// NOLINTEND
