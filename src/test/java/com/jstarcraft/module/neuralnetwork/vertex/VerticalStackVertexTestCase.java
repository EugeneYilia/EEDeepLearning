package com.jstarcraft.module.neuralnetwork.vertex;

import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.impl.StackVertex;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.jstarcraft.module.neuralnetwork.DenseMatrixFactory;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.vertex.transformation.VerticalStackVertex;

public class VerticalStackVertexTestCase extends VertexTestCase {

	@Override
	protected INDArray getError() {
		return Nd4j.linspace(-2.5D, 2.5D, 20).reshape(10, 2);
	}

	@Override
	protected int getSize() {
		return 2;
	}

	@Override
	protected GraphVertex getOldFunction() {
		return new StackVertex(null, "old", 0);
	}

	@Override
	protected Vertex getNewFunction() {
		MatrixFactory cache = new DenseMatrixFactory();
		return new VerticalStackVertex("new", cache);
	}

}
