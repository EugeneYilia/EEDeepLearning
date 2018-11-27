package com.jstarcraft.module.recommendation.evaluator;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.junit.Test;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.DataSplitter;
import com.jstarcraft.module.data.IntegerArray;
import com.jstarcraft.module.data.accessor.AttributeMarker;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.data.convertor.CsvConvertor;
import com.jstarcraft.module.data.processor.DataMatcher;
import com.jstarcraft.module.data.processor.DataSorter;
import com.jstarcraft.module.data.splitter.LeaveOneCrossValidationSplitter;
import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.Recommender;

public abstract class AbstractEvaluatorTestCase<T> {

	protected Configuration configuration;

	protected String userField, itemField, instantField, scoreField;

	protected int userDimension, itemDimension, instantDimension, numberOfUsers, numberOfItems, numberOfInstants;

	protected int[] trainPaginations, trainPositions, testPaginations, testPositions;

	protected SampleAccessor trainMarker, testMarker;

	protected abstract Evaluator<?> getEvaluator(SparseMatrix featureMatrix);

	protected abstract float getMeasure();

	@Test
	public void test() {
		Map<String, Class<?>> discreteFeatures = new HashMap<>();
		Set<String> continuousFeatures = new HashSet<>();
		discreteFeatures.put("user", int.class);
		discreteFeatures.put("item", int.class);
		discreteFeatures.put("instant", long.class);
		continuousFeatures.add("score");
		DataSpace space = new DataSpace(discreteFeatures, continuousFeatures);

		// 制造数据特征
		space.makeFeature("user", "user");
		space.makeFeature("item", "item");
		space.makeFeature("instant", "instant");
		space.makeFeature("score", "score");

		Map<String, String> commands = new HashMap<>();
		commands.put("rec.recommender.ranking.topn", "10");
		configuration = Configuration.valueOf(commands);
		String path = configuration.getString("dfs.data.dir") + "/test/datamodeltest/ratings-date.txt";
		Map<String, Integer> fields = new HashMap<>();
		fields.put("user", 0);
		fields.put("item", 1);
		fields.put("score", 2);
		fields.put("instant", 3);
		CsvConvertor csvConvertor = new CsvConvertor("csv", ' ', path, fields);
		int count = csvConvertor.convert(space);

		// 制造数据模型
		InstanceAccessor model = space.makeModule("model", "user", "item", "instant", "score");

		DataSplitter splitter = new LeaveOneCrossValidationSplitter(model, "user", "instant");
		IntegerArray trainReference = splitter.getTrainReference(0);
		IntegerArray testReference = splitter.getTestReference(0);

		userField = configuration.getString("data.model.dimension.user", "user");
		itemField = configuration.getString("data.model.dimension.item", "item");
		instantField = configuration.getString("data.model.dimension.instant", "instant");
		scoreField = configuration.getString("data.model.dimension.score", "score");

		trainMarker = new AttributeMarker(trainReference, model, scoreField);
		testMarker = new AttributeMarker(testReference, model, scoreField);
		IntegerArray positions = new IntegerArray();
		for (int index = 0, size = model.getSize(); index < size; index++) {
			positions.associateData(index);
		}
		SampleAccessor dataMarker = new AttributeMarker(positions, model, scoreField);

		userDimension = model.getDiscreteDimension(userField);
		itemDimension = model.getDiscreteDimension(itemField);
		instantDimension = model.getDiscreteDimension(instantField);
		numberOfUsers = model.getDiscreteAttribute(userDimension).getSize();
		numberOfItems = model.getDiscreteAttribute(itemDimension).getSize();
		numberOfInstants = model.getDiscreteAttribute(instantDimension).getSize();

		trainPaginations = new int[numberOfUsers + 1];
		trainPositions = new int[trainMarker.getSize()];
		for (int index = 0; index < trainMarker.getSize(); index++) {
			trainPositions[index] = index;
		}
		DataMatcher trainMatcher = DataMatcher.discreteOf(trainMarker, userDimension);
		trainMatcher.match(trainPaginations, trainPositions);
		DataSorter trainSorter = DataSorter.featureOf(trainMarker);
		trainSorter.sort(trainPaginations, trainPositions);
		Table<Integer, Integer, Float> trainTable = HashBasedTable.create();
		for (int position : trainPositions) {
			int rowIndex = trainMarker.getDiscreteFeature(userDimension, position);
			int columnIndex = trainMarker.getDiscreteFeature(itemDimension, position);
			// TODO 处理冲突
			trainTable.put(rowIndex, columnIndex, trainMarker.getMark(position));
		}
		SparseMatrix trainMatrix = SparseMatrix.valueOf(numberOfUsers, numberOfItems, trainTable);

		testPaginations = new int[numberOfUsers + 1];
		testPositions = new int[testMarker.getSize()];
		for (int index = 0; index < testMarker.getSize(); index++) {
			testPositions[index] = index;
		}
		DataMatcher testMatcher = DataMatcher.discreteOf(testMarker, userDimension);
		testMatcher.match(testPaginations, testPositions);
		DataSorter testSorter = DataSorter.featureOf(testMarker);
		testSorter.sort(testPaginations, testPositions);
		Table<Integer, Integer, Float> testTable = HashBasedTable.create();
		for (int position : testPositions) {
			int rowIndex = testMarker.getDiscreteFeature(userDimension, position);
			int columnIndex = testMarker.getDiscreteFeature(itemDimension, position);
			// TODO 处理冲突
			testTable.put(rowIndex, columnIndex, testMarker.getMark(position));
		}
		SparseMatrix testMatrix = SparseMatrix.valueOf(numberOfUsers, numberOfItems, testTable);

		int[] dataPaginations = new int[numberOfUsers + 1];
		int[] dataPositions = new int[dataMarker.getSize()];
		for (int index = 0; index < dataMarker.getSize(); index++) {
			dataPositions[index] = index;
		}
		DataMatcher dataMatcher = DataMatcher.discreteOf(dataMarker, userDimension);
		dataMatcher.match(dataPaginations, dataPositions);
		DataSorter dataSorter = DataSorter.featureOf(dataMarker);
		dataSorter.sort(dataPaginations, dataPositions);
		Table<Integer, Integer, Float> dataTable = HashBasedTable.create();
		for (int position : dataPositions) {
			int rowIndex = dataMarker.getDiscreteFeature(userDimension, position);
			int columnIndex = dataMarker.getDiscreteFeature(itemDimension, position);
			// TODO 处理冲突
			dataTable.put(rowIndex, columnIndex, dataMarker.getMark(position));
		}
		SparseMatrix featureMatrix = SparseMatrix.valueOf(numberOfUsers, numberOfItems, dataTable);

		Recommender recommender = new MockRecommender(itemDimension, trainMatrix);
		Evaluator<?> evaluator = getEvaluator(featureMatrix);
		KeyValue<Integer, Float> sum = evaluate(evaluator, recommender);
		System.out.println(sum.getValue() / sum.getKey());
		Assert.assertThat(sum.getValue() / sum.getKey(), CoreMatchers.equalTo(getMeasure()));
	}

	protected abstract Collection<T> check(int userIndex);

	protected abstract List<KeyValue<Integer, Float>> recommend(Recommender recommender, int userIndex);

	private KeyValue<Integer, Float> evaluate(Evaluator<?> evaluator, Recommender recommender) {
		KeyValue<Integer, Float> sum = new KeyValue<>(0, 0F);
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			// 测试映射
			if (testPaginations[userIndex + 1] - testPaginations[userIndex] == 0) {
				continue;
			}
			// 训练映射
			Collection checkCollection = check(userIndex);
			// 推荐列表
			List<KeyValue<Integer, Float>> recommendList = recommend(recommender, userIndex);
			// 测量列表
			KeyValue<Integer, Float> measure = evaluator.evaluate(checkCollection, recommendList);
			// System.out.println(trainMap.keySet());
			// System.out.println(testMap.keySet());
			// System.out.println(measure.getKey());
			// System.out.println(measure.getValue());
			sum.setKey(sum.getKey() + measure.getKey());
			sum.setValue(sum.getValue() + measure.getValue());
		}
		return sum;
	}

}
