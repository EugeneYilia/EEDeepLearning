package com.jstarcraft.module.neuralnetwork;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.recommendation.recommender.neuralnetwork.ranking.DeepFMOutputConfiguration;
import com.jstarcraft.module.recommendation.recommender.neuralnetwork.ranking.DeepFMProductConfiguration;
import com.jstarcraft.module.recommendation.recommender.neuralnetwork.ranking.DeepFMSumConfiguration;

public class Dl4jTestCase {

	@Test
	public void test() {
		int[] dimensionSizes = new int[] { 2, 3 };

		INDArray dense = Nd4j.create(2, 5);
		dense.put(0, 0, 1D);
		dense.put(0, 3, 1D);
		dense.put(1, 1, 1D);
		dense.put(1, 4, 1D);

		INDArray sparse = Nd4j.create(2, 2);
		sparse.put(0, 0, 0);
		sparse.put(0, 1, 1);
		sparse.put(1, 0, 1);
		sparse.put(1, 1, 2);

		INDArray weight = Nd4j.linspace(0, 9, 10).reshape(5, 2);

		System.out.println(dense.mmul(weight));
		INDArray output = Nd4j.zeros(sparse.rows(), weight.columns());
		for (int row = 0; row < sparse.rows(); row++) {
			for (int column = 0; column < weight.columns(); column++) {
				double value = 0D;
				int cursor = 0;
				for (int index = 0; index < sparse.columns(); index++) {
					value += weight.getDouble(cursor + sparse.getInt(row, index), column);
					cursor += dimensionSizes[index];
				}
				output.put(row, column, value);
			}
		}
		System.out.println(output);

		System.out.println(dense.transpose().mmul(output));
		weight.assign(0);
		for (int index = 0; index < sparse.rows(); index++) {
			for (int column = 0; column < output.columns(); column++) {
				int cursor = 0;
				for (int dimension = 0; dimension < dimensionSizes.length; dimension++) {
					int point = cursor + sparse.getInt(index, dimension);
					double value = weight.getDouble(point, column);
					value += output.getDouble(index, column);
					weight.put(point, column, value);
					cursor += dimensionSizes[dimension];
				}
			}
		}
		System.out.println(weight);
	}

	@Test
	public void testProduct() {
		NeuralNetConfiguration.Builder netBuilder = new NeuralNetConfiguration.Builder();
		// 设置随机种子
		netBuilder.seed(6);
		netBuilder.weightInit(WeightInit.XAVIER_UNIFORM);
		netBuilder.updater(new Sgd(0.01D)).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);

		GraphBuilder graphBuilder = netBuilder.graphBuilder();
		graphBuilder.addInputs("leftInput", "rightInput");
		graphBuilder.addLayer("leftEmbed", new EmbeddingLayer.Builder().nIn(5).nOut(5).activation(Activation.IDENTITY).build(), "leftInput");
		graphBuilder.addLayer("rightEmbed", new EmbeddingLayer.Builder().nIn(5).nOut(5).activation(Activation.IDENTITY).build(), "rightInput");
		graphBuilder.addVertex("product", new DeepFMProductConfiguration(), "leftEmbed", "rightEmbed");
		graphBuilder.addLayer("output", new DeepFMOutputConfiguration.Builder(LossFunctions.LossFunction.SQUARED_LOSS).activation(Activation.IDENTITY).nIn(1).nOut(1).build(), "product");
		graphBuilder.setOutputs("output");

		ComputationGraphConfiguration configuration = graphBuilder.build();

		ComputationGraph graph = new ComputationGraph(configuration);
		graph.init();

		int size = 5;
		INDArray leftData = Nd4j.zeros(size, 1);
		INDArray rightData = Nd4j.zeros(size, 1);
		INDArray labelData = Nd4j.zeros(size, 1).assign(5);
		for (int point = 0; point < 5; point++) {
			leftData.put(point, 0, RandomUtility.randomDouble(5));
			rightData.put(point, 0, RandomUtility.randomDouble(5));
		}

		long time = System.currentTimeMillis();
		System.out.println("testProduct");
		for (int index = 0; index < 50; index++) {
			graph.setInputs(leftData, rightData);
			graph.setLabels(labelData);
			graph.fit();
			System.out.println(graph.score() + " " + graph.outputSingle(leftData, rightData));
		}
		System.out.println(System.currentTimeMillis() - time);
	}

	@Test
	public void testSum() {
		NeuralNetConfiguration.Builder netBuilder = new NeuralNetConfiguration.Builder();
		// 设置随机种子
		netBuilder.seed(6);
		netBuilder.weightInit(WeightInit.XAVIER_UNIFORM);
		netBuilder.updater(new Sgd(0.01D)).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);

		GraphBuilder graphBuilder = netBuilder.graphBuilder();
		graphBuilder.addInputs("leftInput", "rightInput");
		graphBuilder.addLayer("leftEmbed", new EmbeddingLayer.Builder().nIn(5).nOut(5).activation(Activation.IDENTITY).build(), "leftInput");
		graphBuilder.addLayer("rightEmbed", new EmbeddingLayer.Builder().nIn(5).nOut(5).activation(Activation.IDENTITY).build(), "rightInput");
		graphBuilder.addVertex("product", new DeepFMSumConfiguration(), "leftEmbed", "rightEmbed");
		graphBuilder.addLayer("output", new DeepFMOutputConfiguration.Builder(LossFunctions.LossFunction.SQUARED_LOSS).activation(Activation.IDENTITY).nIn(1).nOut(1).build(), "product");
		graphBuilder.setOutputs("output");

		ComputationGraphConfiguration configuration = graphBuilder.build();

		ComputationGraph graph = new ComputationGraph(configuration);
		graph.init();

		int size = 5;
		INDArray leftData = Nd4j.zeros(size, 1);
		INDArray rightData = Nd4j.zeros(size, 1);
		INDArray labelData = Nd4j.zeros(size, 1).assign(5);
		for (int point = 0; point < 5; point++) {
			leftData.put(point, 0, RandomUtility.randomDouble(5));
			rightData.put(point, 0, RandomUtility.randomDouble(5));
		}

		System.out.println("testSum");
		for (int index = 0; index < 50; index++) {
			graph.setInputs(leftData, rightData);
			graph.setLabels(labelData);
			graph.fit();
			System.out.println(graph.score() + " " + graph.outputSingle(leftData, rightData));
		}
	}

}
