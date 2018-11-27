package com.jstarcraft.module.neuralnetwork.parameter;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.model.ModelDefinition;

@ModelDefinition(value = { "copy" })
public class CopyParameterFactory implements ParameterFactory {

	private MathMatrix copy;

	CopyParameterFactory() {
	}

	public CopyParameterFactory(MathMatrix copy) {
		this.copy = copy;
	}

	@Override
	public void setValues(MathMatrix matrix) {
		matrix.mapValues(MatrixMapper.copyOf(copy), null, MathCalculator.SERIAL);
	}

	@Override
	public boolean equals(Object object) {
		if (this == object) {
			return true;
		}
		if (object == null) {
			return false;
		}
		if (getClass() != object.getClass()) {
			return false;
		} else {
			CopyParameterFactory that = (CopyParameterFactory) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.copy, that.copy);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(copy);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "CopyParameterFactory(copy=" + copy + ")";
	}

}
