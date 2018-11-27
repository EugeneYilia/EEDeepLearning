package weka;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Stacking;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class CombineModels {

	public static void main(String[] args) throws Exception {
		// load dataset
		String data = "data/weather.nominal.arff";
		DataSource source = new DataSource(data);
		// get instances object
		Instances dataSet = source.getDataSet();
		dataSet.randomize(new Random(0));
		int trainSize = (int) Math.round(dataSet.numInstances() * 0.8);
		int testSize = dataSet.numInstances() - trainSize;
		Instances trainData = new Instances(dataSet, 0, trainSize);
		Instances testData = new Instances(dataSet, trainSize, testSize);

		// set class index .. as the last attribute
		if (trainData.classIndex() == -1) {
			trainData.setClassIndex(trainData.numAttributes() - 1);
		}
		if (testData.classIndex() == -1) {
			testData.setClassIndex(testData.numAttributes() - 1);
		}

		/*
		 * Boosting a weak classifier using the Adaboost M1 method for boosting
		 * a nominal class classifier Tackles only nominal class problems
		 * Improves performance Sometimes overfits.
		 */
		// AdaBoost ..
		AdaBoostM1 m1 = new AdaBoostM1();
		m1.setClassifier(new DecisionStump());// needs one base-classifier
		m1.setNumIterations(20);
		m1.setDebug(true);
		m1.buildClassifier(trainData);
		System.out.println("booster");
		for (Instance instance : testData) {
			System.out.println(instance.classValue() + ":" + m1.classifyInstance(instance));
		}

		/*
		 * Bagging a classifier to reduce variance. Can do classification and
		 * regression (depending on the base model)
		 */
		// Bagging ..
		Bagging bagger = new Bagging();
		bagger.setClassifier(new RandomTree());// needs one base-model
		bagger.setNumIterations(25);
		bagger.setDebug(true);
		bagger.buildClassifier(trainData);
		System.out.println("bagger");
		for (Instance instance : testData) {
			System.out.println(instance.classValue() + ":" + bagger.classifyInstance(instance));
		}

		/*
		 * The Stacking method combines several models Can do classification or
		 * regression.
		 */
		// Stacking ..
		Stacking stacker = new Stacking();
		stacker.setMetaClassifier(new J48());// needs one meta-model
		Classifier[] classifiers = { new J48(), new NaiveBayes(), new RandomForest() };
		stacker.setClassifiers(classifiers);// needs one or more models
		stacker.setDebug(true);
		stacker.buildClassifier(trainData);
		System.out.println("stacker");
		for (Instance instance : testData) {
			System.out.println(instance.classValue() + ":" + stacker.classifyInstance(instance));
		}

		/*
		 * Class for combining classifiers. Different combinations of
		 * probability estimates for classification are available.
		 */
		// Vote ..
		Vote voter = new Vote();
		voter.setClassifiers(classifiers);// needs one or more classifiers
		voter.setDebug(true);
		voter.buildClassifier(trainData);
		System.out.println("voter");
		for (Instance instance : testData) {
			System.out.println(instance.classValue() + ":" + voter.classifyInstance(instance));
		}
	}

}