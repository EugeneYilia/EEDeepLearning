package com.jstarcraft.module.neuralnetwork.vertex;

import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.jstarcraft.module.neuralnetwork.DenseMatrixFactory;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.vertex.accumulation.SumVertex;
import com.jstarcraft.module.recommendation.recommender.neuralnetwork.ranking.DeepFMSumVertex;

public class SumVertexTestCase extends VertexTestCase {

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
		return new DeepFMSumVertex(null, "old", 0);
	}

	@Override
	protected Vertex getNewFunction() {
		MatrixFactory cache = new DenseMatrixFactory();
		return new SumVertex("new", cache);
	}

}
