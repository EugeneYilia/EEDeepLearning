package com.jstarcraft.module.math.structure.tensor;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.junit.Assert;
import org.junit.Test;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.IntegerArray;
import com.jstarcraft.module.data.accessor.AttributeMarker;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.data.convertor.CsvConvertor;
import com.jstarcraft.module.data.processor.DataMatcher;
import com.jstarcraft.module.data.processor.DataSorter;
import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;

public class SparseTensorTestCase {

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

		Configuration configuration = Configuration.valueOf();
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
		String userField = configuration.getString("data.model.dimension.user", "user");
		String itemField = configuration.getString("data.model.dimension.item", "item");
		String instantField = configuration.getString("data.model.dimension.instant", "instant");
		String scoreField = configuration.getString("data.model.dimension.score", "score");
		int userDimension = model.getDiscreteDimension(userField);
		int itemDimension = model.getDiscreteDimension(itemField);
		int instantDimension = model.getDiscreteDimension(instantField);
		int numberOfUsers = model.getDiscreteAttribute(userDimension).getSize();
		int numberOfItems = model.getDiscreteAttribute(itemDimension).getSize();
		int numberOfInstants = model.getDiscreteAttribute(instantDimension).getSize();
		IntegerArray positions = new IntegerArray();
		for (int index = 0, size = model.getSize(); index < size; index++) {
			positions.associateData(index);
		}
		SampleAccessor dataMarker = new AttributeMarker(positions, model, scoreField);
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

		// 检查张量的存储排序
		int[] dimensions = new int[] { numberOfUsers, numberOfItems, numberOfInstants };
		SparseTensor featureTensor = SparseTensor.valueOf(dimensions, dataMarker, (keyValue, instance) -> {
			int indexes[] = keyValue.getKey();
			for (int dimension = 0; dimension < indexes.length; dimension++) {
				indexes[dimension] = instance.getDiscreteFeature(dimension);
			}
			keyValue.setValue(instance.getMark());
		});
		SparseMatrix tensorMatrix = featureTensor.toMatrix(userDimension, itemDimension);

		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			Iterator<VectorScalar> leftIterator = featureMatrix.getRowVector(userIndex).iterator();
			Iterator<VectorScalar> rightIterator = tensorMatrix.getRowVector(userIndex).iterator();
			while (leftIterator.hasNext() && rightIterator.hasNext()) {
				VectorScalar leftTerm = leftIterator.next();
				VectorScalar rightTerm = rightIterator.next();
				if (leftTerm.getValue() != rightTerm.getValue()) {
					System.out.println(dataMarker.getDiscreteAttribute(userDimension).getDatas()[userIndex]);
					System.out.println(dataMarker.getDiscreteAttribute(itemDimension).getDatas()[leftTerm.getIndex()]);
					System.out.println(userIndex);
					System.out.println(leftTerm.getIndex());
					System.out.println(leftTerm.getValue());
					System.out.println(dataMarker.getDiscreteAttribute(userDimension).getDatas()[userIndex]);
					System.out.println(dataMarker.getDiscreteAttribute(itemDimension).getDatas()[rightTerm.getIndex()]);
					System.out.println(userIndex);
					System.out.println(rightTerm.getIndex());
					System.out.println(rightTerm.getValue());
					Assert.fail();
				}
			}
		}

	}

}
