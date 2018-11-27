package com.jstarcraft.module.neuralnetwork.vertex;

import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.impl.L2Vertex;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.jstarcraft.module.neuralnetwork.DenseMatrixFactory;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.vertex.accumulation.EuclideanVertex;

public class EuclideanVertexTestCase extends VertexTestCase {

	@Override
	protected INDArray getError() {
		return Nd4j.linspace(-2.5D, 2.5D, 5).reshape(5, 1);
	}

	@Override
	protected int getSize() {
		return 2;
	}

	@Override
	protected GraphVertex getOldFunction() {
		return new L2Vertex(null, "old", 0, EuclideanVertex.DEFAULT_EPSILON);
	}

	@Override
	protected Vertex getNewFunction() {
		MatrixFactory cache = new DenseMatrixFactory();
		return new EuclideanVertex("new", cache);
	}

}
