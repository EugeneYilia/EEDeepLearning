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
package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.NeuralNetworkRecommender;

/**
 * Yao et al., <strong>Collaborative Denoising Auto-Encoders for Top-N
 * Recommender Systems, WSDM 2016.
 *
 * @author Ma Chen
 */

public class CDAERecommender extends NeuralNetworkRecommender {

	/**
	 * the threshold to binarize the rating
	 */
	private double binarie;

	@Override
	protected int getInputDimension() {
		return numberOfItems;
	}

	@Override
	protected MultiLayerConfiguration getNetworkConfiguration() {
		NeuralNetConfiguration.ListBuilder factory = new NeuralNetConfiguration.Builder().seed(6)
				// .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
				// .gradientNormalizationThreshold(1.0)
				.updater(new Nesterovs(learnRate, momentum)).weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).l2(weightRegularization).list();
		factory.layer(0, new CDAEConfiguration.Builder().nIn(inputDimension).nOut(hiddenDimension).activation(Activation.fromString(hiddenActivation)).setNumUsers(numberOfUsers).build());
		factory.layer(1, new OutputLayer.Builder().nIn(hiddenDimension).nOut(inputDimension).lossFunction(LossFunctions.LossFunction.SQUARED_LOSS).activation(Activation.fromString(outputActivation)).build());
		factory.pretrain(false).backprop(true);
		MultiLayerConfiguration configuration = factory.build();
		return configuration;
	}

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		binarie = configuration.getDouble("rec.binarize.threshold");
		// transform the sparse matrix to INDArray
		// the sparse training matrix has been binarized

		int[] matrixShape = new int[] { numberOfUsers, numberOfItems };
		inputData = Nd4j.zeros(matrixShape);
		for (MatrixScalar term : trainMatrix) {
			if (term.getValue() > binarie) {
				inputData.putScalar(term.getRow(), term.getColumn(), 1D);
			}
		}
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		return outputData.getFloat(userIndex, itemIndex);
	}

}
