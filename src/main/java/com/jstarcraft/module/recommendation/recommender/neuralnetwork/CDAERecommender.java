/**
 * Copyright (C) 2016 LibRec
 * <p>
 * This file is part of LibRec.
 * LibRec is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * <p>
 * LibRec is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License
 * along with LibRec. If not, see <http://www.gnu.org/licenses/>.
 */
package com.jstarcraft.module.recommendation.recommender.neuralnetwork;

import java.util.HashMap;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.matrix.Nd4jMatrix;
import com.jstarcraft.module.neuralnetwork.Graph;
import com.jstarcraft.module.neuralnetwork.GraphConfigurator;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.Nd4jMatrixFactory;
import com.jstarcraft.module.neuralnetwork.activation.IdentityActivationFunction;
import com.jstarcraft.module.neuralnetwork.activation.SigmoidActivationFunction;
import com.jstarcraft.module.neuralnetwork.layer.Layer;
import com.jstarcraft.module.neuralnetwork.layer.Layer.Mode;
import com.jstarcraft.module.neuralnetwork.layer.ParameterConfigurator;
import com.jstarcraft.module.neuralnetwork.layer.WeightLayer;
import com.jstarcraft.module.neuralnetwork.learn.NesterovLearner;
import com.jstarcraft.module.neuralnetwork.loss.MSELossFunction;
import com.jstarcraft.module.neuralnetwork.normalization.IgnoreNormalizer;
import com.jstarcraft.module.neuralnetwork.optimization.StochasticGradientOptimizer;
import com.jstarcraft.module.neuralnetwork.parameter.XavierParameterFactory;
import com.jstarcraft.module.neuralnetwork.schedule.ConstantSchedule;
import com.jstarcraft.module.neuralnetwork.vertex.LayerVertex;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.ModelRecommender;

/**
 * Yao et al., <strong>Collaborative Denoising Auto-Encoders for Top-N
 * Recommender Systems, WSDM 2016.
 *
 * @author Ma Chen
 */

public class CDAERecommender extends ModelRecommender {

	/**
	 * the dimension of input units
	 */
	protected int inputDimension;

	/**
	 * the dimension of hidden units
	 */
	protected int hiddenDimension;

	/**
	 * the activation function of the hidden layer in the neural network
	 */
	protected String hiddenActivation;

	/**
	 * the activation function of the output layer in the neural network
	 */
	protected String outputActivation;

	/**
	 * the learning rate of the optimization algorithm
	 */
	protected float learnRate;

	/**
	 * the momentum of the optimization algorithm
	 */
	protected float momentum;

	/**
	 * the regularization coefficient of the weights in the neural network
	 */
	protected float weightRegularization;

	/**
	 * the data structure that stores the training data
	 */
	protected Nd4jMatrix inputData;

	protected Nd4jMatrix labelData;

	/**
	 * the data structure that stores the predicted data
	 */
	protected Nd4jMatrix outputData;

	protected Graph network;

	/**
	 * the threshold to binarize the rating
	 */
	private float binarie;

	protected int getInputDimension() {
		return numberOfItems;
	}

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		inputDimension = getInputDimension();
		hiddenDimension = configuration.getInteger("rec.hidden.dimension");
		hiddenActivation = configuration.getString("rec.hidden.activation");
		outputActivation = configuration.getString("rec.output.activation");
		learnRate = configuration.getFloat("rec.iterator.learnrate");
		momentum = configuration.getFloat("rec.iterator.momentum");
		weightRegularization = configuration.getFloat("rec.weight.regularization");
		binarie = configuration.getFloat("rec.binarize.threshold");
		// transform the sparse matrix to INDArray
		// the sparse training matrix has been binarized

		INDArray array = Nd4j.create(numberOfUsers, numberOfItems);
		inputData = new Nd4jMatrix(array);

		array = Nd4j.create(numberOfUsers, numberOfItems);
		labelData = new Nd4jMatrix(array);
		for (MatrixScalar term : trainMatrix) {
			labelData.setValue(term.getRow(), term.getColumn(), 1F);
		}

		array = Nd4j.create(numberOfUsers, numberOfItems);
		outputData = new Nd4jMatrix(array);
	}

	protected Graph getComputationGraph() {
		GraphConfigurator configurator = new GraphConfigurator();
		Map<String, ParameterConfigurator> configurators = new HashMap<>();
		Nd4j.getRandom().setSeed(6L);
		ParameterConfigurator parameterConfigurator = new ParameterConfigurator(0F, weightRegularization, new XavierParameterFactory());
		configurators.put(CDAELayer.WEIGHT_KEY, parameterConfigurator);
		configurators.put(CDAELayer.BIAS_KEY, new ParameterConfigurator(0F, 0F));
		configurators.put(CDAELayer.USER_KEY, parameterConfigurator);
		MatrixFactory factory = new Nd4jMatrixFactory();
		Layer cdaeLayer = new CDAELayer(numberOfUsers, numberOfItems, hiddenDimension, factory, configurators, Mode.TRAIN, new SigmoidActivationFunction());
		Layer outputLayer = new WeightLayer(hiddenDimension, numberOfItems, factory, configurators, Mode.TRAIN, new IdentityActivationFunction());

		configurator.connect(new LayerVertex("cdae", factory, cdaeLayer, new NesterovLearner(new ConstantSchedule(momentum), new ConstantSchedule(learnRate)), new IgnoreNormalizer()));
		configurator.connect(new LayerVertex("output", factory, outputLayer, new NesterovLearner(new ConstantSchedule(momentum), new ConstantSchedule(learnRate)), new IgnoreNormalizer()), "cdae");

		Graph graph = new Graph(configurator, new StochasticGradientOptimizer(), new MSELossFunction());
		return graph;
	}

	@Override
	protected void doPractice() {
		Graph graph = getComputationGraph();
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			inputData.getArray().assign(labelData.getArray());
			for (MatrixScalar term : trainMatrix) {
				if (RandomUtility.randomFloat(1F) < 0.2F) {
					inputData.setValue(term.getRow(), term.getColumn(), 0F);
				}
			}
			totalLoss = graph.practice(1, new MathMatrix[] { inputData }, new MathMatrix[] { labelData });
			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			currentLoss = totalLoss;
		}
		graph.predict(new MathMatrix[] { labelData }, new MathMatrix[] { outputData });
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		return outputData.getValue(userIndex, itemIndex);
	}

}
