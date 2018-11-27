package com.jstarcraft.module.neuralnetwork.layer;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.neuralnetwork.DenseMatrixFactory;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.activation.SigmoidActivationFunction;
import com.jstarcraft.module.neuralnetwork.layer.Layer.Mode;
import com.jstarcraft.module.neuralnetwork.parameter.CopyParameterFactory;
import com.jstarcraft.module.recommendation.recommender.neuralnetwork.FMLayer;
import com.jstarcraft.module.recommendation.recommender.neuralnetwork.ranking.DeepFMInputConfiguration;

public class FMLayerTestCase extends LayerTestCase {

	@Override
	protected INDArray getData() {
		return Nd4j.linspace(0, 9, 10).reshape(5, 2);
	}

	@Override
	protected INDArray getError() {
		return Nd4j.linspace(-2.5D, 2.5D, 5).reshape(5, 1);
	}

	@Override
	protected List<KeyValue<String, String>> getGradients() {
		List<KeyValue<String, String>> gradients = new LinkedList<>();
		gradients.add(new KeyValue<>(DefaultParamInitializer.WEIGHT_KEY, FMLayer.WEIGHT_KEY));
		gradients.add(new KeyValue<>(DefaultParamInitializer.BIAS_KEY, FMLayer.BIAS_KEY));
		return gradients;
	}

	@Override
	protected AbstractLayer<?> getOldFunction() {
		NeuralNetConfiguration neuralNetConfiguration = new NeuralNetConfiguration();
		DeepFMInputConfiguration layerConfiguration = new DeepFMInputConfiguration(new int[] { 10, 10 });
		layerConfiguration.setWeightInit(WeightInit.UNIFORM);
		layerConfiguration.setNOut(1);
		layerConfiguration.setActivationFn(new ActivationSigmoid());
		layerConfiguration.setL1(0.01D);
		layerConfiguration.setL1Bias(0.01D);
		layerConfiguration.setL2(0.05D);
		layerConfiguration.setL2Bias(0.05D);
		neuralNetConfiguration.setLayer(layerConfiguration);
		AbstractLayer<?> layer = AbstractLayer.class.cast(layerConfiguration.instantiate(neuralNetConfiguration, null, 0, Nd4j.zeros(21), true));
		layer.setBackpropGradientsViewArray(Nd4j.zeros(21));
		return layer;
	}

	@Override
	protected Layer getNewFunction(AbstractLayer<?> layer) {
		Map<String, ParameterConfigurator> configurators = new HashMap<>();
		CopyParameterFactory weight = new CopyParameterFactory(getMatrix(layer.getParam(DefaultParamInitializer.WEIGHT_KEY)));
		configurators.put(WeightLayer.WEIGHT_KEY, new ParameterConfigurator(0.01F, 0.05F, weight));
		CopyParameterFactory bias = new CopyParameterFactory(getMatrix(layer.getParam(DefaultParamInitializer.BIAS_KEY)));
		configurators.put(WeightLayer.BIAS_KEY, new ParameterConfigurator(0.01F, 0.05F, bias));
		MatrixFactory factory = new DenseMatrixFactory();
		return new FMLayer(new int[] { 10, 10 }, 20, 1, factory, configurators, Mode.TRAIN, new SigmoidActivationFunction());
	}

}
