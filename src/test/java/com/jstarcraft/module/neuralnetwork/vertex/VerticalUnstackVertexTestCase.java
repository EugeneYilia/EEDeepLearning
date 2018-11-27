package com.jstarcraft.module.neuralnetwork.vertex;

import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.impl.UnstackVertex;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.RowCompositeMatrix;
import com.jstarcraft.module.neuralnetwork.DenseMatrixFactory;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.vertex.transformation.VerticalUnstackVertex;

public class VerticalUnstackVertexTestCase extends VertexTestCase {

	@Override
	protected MathMatrix getMatrix(INDArray array) {
		int size = array.rows();
		MathMatrix[] components = new MathMatrix[size];
		for (int index = 0; index < size; index++) {
			components[index] = DenseMatrix.valueOf(1, array.columns());
		}
		MathMatrix matrix = RowCompositeMatrix.attachOf(components);
		matrix.mapValues((row, column, value, message) -> {
			return array.getFloat(row, column);
		}, null, MathCalculator.SERIAL);
		return matrix;
	}

	@Override
	protected INDArray getError() {
		return Nd4j.linspace(-2.5D, 2.5D, 4).reshape(2, 2);
	}

	@Override
	protected int getSize() {
		return 1;
	}

	@Override
	protected GraphVertex getOldFunction() {
		return new UnstackVertex(null, "old", 0, 1, 2);
	}

	@Override
	protected Vertex getNewFunction() {
		MatrixFactory cache = new DenseMatrixFactory();
		return new VerticalUnstackVertex("new", cache, 2, 4);
	}

}
