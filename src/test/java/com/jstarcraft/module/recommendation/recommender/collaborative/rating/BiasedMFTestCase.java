package com.jstarcraft.module.recommendation.recommender.collaborative.rating;

import java.util.Map;

import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.junit.Test;

import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.evaluator.rating.MAEEvaluator;
import com.jstarcraft.module.recommendation.evaluator.rating.MPEEvaluator;
import com.jstarcraft.module.recommendation.evaluator.rating.MSEEvaluator;
import com.jstarcraft.module.recommendation.task.RatingTask;

public class BiasedMFTestCase {

	@Test
	public void testRecommender() throws Exception {
		Configuration configuration = Configuration.valueOf("rec/collaborative/rating/biasedmf-test.properties");
		RatingTask job = new RatingTask(configuration);
		Map<String, Float> measures = job.execute();
		Assert.assertThat(measures.get(MAEEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.62598F));
		Assert.assertThat(measures.get(MPEEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.9902817F));
		Assert.assertThat(measures.get(MSEEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.649835F));
	}

}
