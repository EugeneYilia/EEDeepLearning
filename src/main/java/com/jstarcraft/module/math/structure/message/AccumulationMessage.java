package com.jstarcraft.module.math.structure.message;

import com.jstarcraft.module.math.structure.MathMessage;

@Deprecated
public interface AccumulationMessage<T> extends MathMessage<T> {

	void accumulateValue(float value);

	T getValue();

}
