package com.jstarcraft.module.similarity;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.junit.Test;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.IntegerArray;
import com.jstarcraft.module.data.accessor.AttributeMarker;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.data.convertor.CsvConvertor;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.math.structure.matrix.SymmetryMatrix;
import com.jstarcraft.module.math.structure.tensor.SparseTensor;
import com.jstarcraft.module.recommendation.configure.Configuration;

public abstract class AbstractSimilarityTestCase {

	protected abstract boolean checkCorrelation(float correlation);

	protected abstract float getIdentical();

	protected abstract Similarity getSimilarity();

	@Test
	public void test() {
		Map<String, Class<?>> discreteFeatures = new HashMap<>();
		Set<String> continuousFeatures = new HashSet<>();
		discreteFeatures.put("user", int.class);
		discreteFeatures.put("item", int.class);
		continuousFeatures.add("score");
		DataSpace space = new DataSpace(discreteFeatures, continuousFeatures);

		// 制造数据特征
		space.makeFeature("user", "user");
		space.makeFeature("item", "item");
		space.makeFeature("score", "score");

		Configuration configuration = Configuration.valueOf();
		String path = configuration.getString("dfs.data.dir") + "/filmtrust/rating";
		Map<String, Integer> fields = new HashMap<>();
		fields.put("user", 0);
		fields.put("item", 1);
		fields.put("score", 2);
		CsvConvertor csvConvertor = new CsvConvertor("csv", ' ', path, fields);
		int count = csvConvertor.convert(space);

		// 制造数据模型
		InstanceAccessor model = space.makeModule("model", "user", "item", "score");

		String userField = configuration.getString("data.model.fields.user");
		String itemField = configuration.getString("data.model.fields.item");
		String scoreField = configuration.getString("data.model.fields.score");

		IntegerArray positions = new IntegerArray();
		for (int index = 0, size = model.getSize(); index < size; index++) {
			positions.associateData(index);
		}
		SampleAccessor marker = new AttributeMarker(positions, model, scoreField);

		int userDimension = marker.getDiscreteDimension(userField);
		int itemDimension = marker.getDiscreteDimension(itemField);
		int[] dimensions = new int[] { marker.getDiscreteAttribute(userDimension).getSize(), marker.getDiscreteAttribute(itemDimension).getSize() };
		SparseTensor scoreTensor = SparseTensor.valueOf(dimensions, marker, (keyValue, instance) -> {
			int indexes[] = keyValue.getKey();
			for (int dimension = 0; dimension < indexes.length; dimension++) {
				indexes[dimension] = instance.getDiscreteFeature(dimension);
			}
			keyValue.setValue(instance.getMark());
		});
		SparseMatrix scoreMatrix = scoreTensor.toMatrix(userDimension, itemDimension);

		Similarity similarity = getSimilarity();
		SymmetryMatrix similarityMatrix = similarity.makeSimilarityMatrix(scoreMatrix, false, configuration.getFloat("rec.similarity.shrinkage", 0F));
		assertEquals(space.getDiscreteAttribute(userField).getSize(), similarityMatrix.getRowSize());
		for (MatrixScalar term : similarityMatrix) {
			assertTrue(checkCorrelation(term.getValue()));
		}
		for (int index = 0, size = space.getDiscreteAttribute(userField).getSize(); index < size; index++) {
			Assert.assertThat(similarityMatrix.getValue(index, index), CoreMatchers.equalTo(getIdentical()));
		}

		similarityMatrix = similarity.makeSimilarityMatrix(scoreMatrix, true, configuration.getFloat("rec.similarity.shrinkage", 0F));
		assertEquals(space.getDiscreteAttribute(itemField).getSize(), similarityMatrix.getRowSize());
		for (MatrixScalar term : similarityMatrix) {
			assertTrue(checkCorrelation(term.getValue()));
		}
		for (int index = 0, size = space.getDiscreteAttribute(itemField).getSize(); index < size; index++) {
			Assert.assertThat(similarityMatrix.getValue(index, index), CoreMatchers.equalTo(getIdentical()));
		}
	}

}
