package com.jstarcraft.module.recommendation.task;

import java.lang.reflect.Type;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.jstarcraft.core.utility.JsonUtility;
import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.core.utility.ReflectionUtility;
import com.jstarcraft.core.utility.StringUtility;
import com.jstarcraft.core.utility.TypeUtility;
import com.jstarcraft.module.data.DataConvertor;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.DataSplitter;
import com.jstarcraft.module.data.IntegerArray;
import com.jstarcraft.module.data.accessor.AttributeMarker;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.data.processor.DataMatcher;
import com.jstarcraft.module.data.processor.DataSorter;
import com.jstarcraft.module.data.splitter.GivenDataSplitter;
import com.jstarcraft.module.data.splitter.GivenNumberSplitter;
import com.jstarcraft.module.data.splitter.KFoldCrossValidationSplitter;
import com.jstarcraft.module.data.splitter.LeaveOneCrossValidationSplitter;
import com.jstarcraft.module.data.splitter.RandomSplitter;
import com.jstarcraft.module.data.splitter.RatioSplitter;
import com.jstarcraft.module.environment.EnvironmentContext;
import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.evaluator.Evaluator;
import com.jstarcraft.module.recommendation.exception.RecommendationException;
import com.jstarcraft.module.recommendation.recommender.Recommender;
import com.jstarcraft.module.recommendation.utility.DriverUtility;

public abstract class AbstractTask<T> {

	protected final Logger logger = LoggerFactory.getLogger(this.getClass());

	protected Configuration configuration;

	protected String userField, itemField, scoreField;

	protected int userDimension, itemDimension, numberOfUsers, numberOfItems;

	protected int[] trainPaginations, trainPositions, testPaginations, testPositions;

	protected SampleAccessor dataMarker, trainMarker, testMarker;

	protected Recommender recommender;

	protected AbstractTask(Configuration configuration) {
		this.configuration = configuration;
		Long seed = configuration.getLong("rec.random.seed");
		if (seed != null) {
			RandomUtility.setSeed(seed);
		}
		this.recommender = (Recommender) ReflectionUtility.getInstance((Class<Recommender>) DriverUtility.getClass(configuration.getString("rec.recommender.class")));
	}

	protected abstract Collection<Evaluator> getEvaluators(SparseMatrix featureMatrix);

	protected abstract Collection<T> check(int userIndex);

	protected abstract List<KeyValue<Integer, Float>> recommend(Recommender recommender, int userIndex);

	private ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

	private Map<Class<? extends Evaluator>, KeyValue<Integer, Float>> evaluate(Collection<Evaluator> evaluators, Recommender recommender) {
		Map<Class<? extends Evaluator>, KeyValue<Integer, Float>[]> values = new HashMap<>();
		for (Evaluator evaluator : evaluators) {
			values.put(evaluator.getClass(), new KeyValue[numberOfUsers]);
		}
		// 按照用户切割任务.
		CountDownLatch latch = new CountDownLatch(numberOfUsers);
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			int index = userIndex;
			executor.submit(() -> {
				if (testPaginations[index + 1] - testPaginations[index] != 0) {
					// 校验集合
					Collection checkCollection = check(index);
					// 推荐列表
					List<KeyValue<Integer, Float>> recommendList = recommend(recommender, index);
					// 测量列表
					for (Evaluator<?> evaluator : evaluators) {
						KeyValue<Integer, Float>[] measures = values.get(evaluator.getClass());
						KeyValue<Integer, Float> measure = evaluator.evaluate(checkCollection, recommendList);
						measures[index] = measure;
					}
				}
				latch.countDown();
			});
		}
		try {
			latch.await();
		} catch (Exception exception) {
			throw new RecommendationException(exception);
		}

		Map<Class<? extends Evaluator>, KeyValue<Integer, Float>> measures = new HashMap<>();
		for (Entry<Class<? extends Evaluator>, KeyValue<Integer, Float>[]> term : values.entrySet()) {
			KeyValue<Integer, Float> measure = new KeyValue<>(0, 0F);
			// if (term.getKey() == RecallEvaluator.class) {
			// for (KeyValue<Integer, Double> element : term.getValue()) {
			// if (element == null) {
			// continue;
			// }
			// System.out.println(element.getKey() + " " + element.getValue());
			// }
			// }
			for (KeyValue<Integer, Float> element : term.getValue()) {
				if (element == null) {
					continue;
				}
				measure.setKey(measure.getKey() + element.getKey());
				measure.setValue(measure.getValue() + element.getValue());
			}
			measures.put(term.getKey(), measure);
		}
		return measures;
	}

	public Map<String, Float> execute() throws Exception {
		// TODO 数据属性部分
		// 离散属性
		Type dicreteConfiguration = TypeUtility.parameterize(HashMap.class, String.class, Class.class);
		Map<String, Class<?>> dicreteDifinitions = JsonUtility.string2Object(configuration.getString("data.space.attributes.dicrete"), dicreteConfiguration);
		// 连续属性
		Type continuousConfiguration = TypeUtility.parameterize(HashSet.class, String.class);
		Set<String> continuousDifinitions = JsonUtility.string2Object(configuration.getString("data.space.attributes.continuous"), continuousConfiguration);

		// TODO 数据特征部分
		Type featureConfiguration = TypeUtility.parameterize(HashMap.class, String.class, String.class);
		Map<String, String> featureDifinitions = JsonUtility.string2Object(configuration.getString("data.space.features"), featureConfiguration);

		// 数据空间部分
		DataSpace space = new DataSpace(dicreteDifinitions, continuousDifinitions);
		for (Entry<String, String> term : featureDifinitions.entrySet()) {
			space.makeFeature(term.getKey(), term.getValue());
		}

		// TODO 数据转换器部分
		Map<String, Integer> counts = new HashMap<>();
		String format = configuration.getString("data.format");
		Type convertorConfiguration = TypeUtility.parameterize(LinkedHashMap.class, String.class, TypeUtility.parameterize(KeyValue.class, String.class, HashMap.class));
		Map<String, KeyValue<String, HashMap<String, ?>>> convertorDifinitions = JsonUtility.string2Object(configuration.getString("data.convertors"), convertorConfiguration);
		for (Entry<String, KeyValue<String, HashMap<String, ?>>> term : convertorDifinitions.entrySet()) {
			String name = term.getKey();
			KeyValue<String, HashMap<String, ?>> keyValue = term.getValue();
			DataConvertor convertor = null;
			switch (format) {
			case "arff": {
				convertor = ReflectionUtility.getInstance((Class<DataConvertor>) DriverUtility.getClass(format), name, keyValue.getKey(), keyValue.getValue());
				break;
			}
			case "csv": {
				convertor = ReflectionUtility.getInstance((Class<DataConvertor>) DriverUtility.getClass(format), name, configuration.getCharacter("data.splitter.delimiter", ' '), keyValue.getKey(), keyValue.getValue());
				break;
			}
			default: {
				throw new RecommendationException("不支持的转换格式");
			}
			}
			counts.put(name, convertor.convert(space));
		}

		// TODO 数据模型部分
		Type modelConfiguration = TypeUtility.parameterize(HashMap.class, String.class, String[].class);
		Map<String, String[]> modelDifinitions = JsonUtility.string2Object(configuration.getString("data.models"), modelConfiguration);
		for (Entry<String, String[]> term : modelDifinitions.entrySet()) {
			space.makeModule(term.getKey(), term.getValue());
		}

		// TODO 数据切割器部分
		SplitConfiguration splitterDifinition = JsonUtility.string2Object(configuration.getString("data.splitter"), SplitConfiguration.class);
		InstanceAccessor model = space.getModule(splitterDifinition.model);
		DataSplitter splitter;
		switch (splitterDifinition.type) {
		case "kcv": {
			int size = configuration.getInteger("data.splitter.kcv.number", 1);
			splitter = new KFoldCrossValidationSplitter(model, size);
			break;
		}
		case "loocv": {
			splitter = new LeaveOneCrossValidationSplitter(model, splitterDifinition.matchField, splitterDifinition.sortField);
			break;
		}
		case "testset": {
			String name = configuration.getString("data.splitter.given-data.name");
			splitter = new GivenDataSplitter(model, counts.get(name));
			break;
		}
		case "givenn": {
			int number = configuration.getInteger("data.splitter.given-number.number");
			splitter = new GivenNumberSplitter(model, splitterDifinition.matchField, splitterDifinition.sortField, number);
			break;
		}
		case "random": {
			double random = configuration.getDouble("data.splitter.random.value", 0.8D);
			splitter = new RandomSplitter(model, splitterDifinition.matchField, random);
			break;
		}
		case "ratio": {
			double ratio = configuration.getDouble("data.splitter.ratio.value", 0.8D);
			splitter = new RatioSplitter(model, splitterDifinition.matchField, splitterDifinition.sortField, ratio);
			break;
		}
		default: {
			throw new RecommendationException("不支持的划分类型");
		}
		}

		// 评估部分
		userField = configuration.getString("data.model.fields.user", "user");
		itemField = configuration.getString("data.model.fields.item", "item");
		scoreField = configuration.getString("data.model.fields.score", "score");

		Double binarize = configuration.getDouble("data.convert.binarize.threshold");
		Map<String, Float> measures = new TreeMap<>();

		EnvironmentContext context = Nd4j.getAffinityManager().getClass().getSimpleName().equals("CpuAffinityManager") ? EnvironmentContext.CPU : EnvironmentContext.GPU;
		Future<?> task = context.doTask(() -> {
			for (int index = 0; index < splitter.getSize(); index++) {
				IntegerArray trainReference = splitter.getTrainReference(index);
				IntegerArray testReference = splitter.getTestReference(index);
				trainMarker = new AttributeMarker(trainReference, space.getModule(splitterDifinition.model), scoreField);
				testMarker = new AttributeMarker(testReference, space.getModule(splitterDifinition.model), scoreField);

				IntegerArray positions = new IntegerArray();
				for (int position = 0, size = model.getSize(); position < size; position++) {
					positions.associateData(position);
				}
				dataMarker = new AttributeMarker(positions, model, scoreField);

				userDimension = model.getDiscreteDimension(userField);
				itemDimension = model.getDiscreteDimension(itemField);
				numberOfUsers = model.getDiscreteAttribute(userDimension).getSize();
				numberOfItems = model.getDiscreteAttribute(itemDimension).getSize();

				trainPaginations = new int[numberOfUsers + 1];
				trainPositions = new int[trainMarker.getSize()];
				for (int position = 0, size = trainMarker.getSize(); position < size; position++) {
					trainPositions[position] = position;
				}
				DataMatcher trainMatcher = DataMatcher.discreteOf(trainMarker, userDimension);
				trainMatcher.match(trainPaginations, trainPositions);

				testPaginations = new int[numberOfUsers + 1];
				testPositions = new int[testMarker.getSize()];
				for (int position = 0, size = testMarker.getSize(); position < size; position++) {
					testPositions[position] = position;
				}
				DataMatcher testMatcher = DataMatcher.discreteOf(testMarker, userDimension);
				testMatcher.match(testPaginations, testPositions);

				int[] dataPaginations = new int[numberOfUsers + 1];
				int[] dataPositions = new int[dataMarker.getSize()];
				for (int position = 0; position < dataMarker.getSize(); position++) {
					dataPositions[position] = position;
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

				recommender.prepare(configuration, trainMarker, model, space);
				recommender.practice();
				for (Entry<Class<? extends Evaluator>, KeyValue<Integer, Float>> measure : evaluate(getEvaluators(featureMatrix), recommender).entrySet()) {
					Float value = measure.getValue().getValue() / measure.getValue().getKey();
					measures.put(measure.getKey().getSimpleName(), value);
				}
			}
		});
		task.get();

		for (Entry<String, Float> term : measures.entrySet()) {
			term.setValue(term.getValue() / splitter.getSize());
			if (logger.isInfoEnabled()) {
				logger.info(StringUtility.format("measure of {} is {}", term.getKey(), term.getValue()));
			}
		}
		return measures;
	}

	public Recommender getRecommender() {
		return recommender;
	}

	private static class SplitConfiguration {

		private String model;

		private String type;

		private String matchField;

		private String sortField;

	}

}
