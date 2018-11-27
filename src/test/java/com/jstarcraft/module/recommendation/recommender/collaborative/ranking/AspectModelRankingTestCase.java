package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import java.util.Map;

import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.junit.Test;

import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.evaluator.ranking.AUCEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.MAPEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.NDCGEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.NoveltyEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.PrecisionEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.RecallEvaluator;
import com.jstarcraft.module.recommendation.task.RankingTask;
import com.jstarcraft.module.recommendation.evaluator.ranking.MRREvaluator;

public class AspectModelRankingTestCase {

	@Test
	public void testRecommender() throws Exception {
		Configuration configuration = Configuration.valueOf("rec/collaborative/ranking/aspectmodelranking-test.properties");
		RankingTask job = new RankingTask(configuration);
		Map<String, Float> measures = job.execute();
		Assert.assertThat(measures.get(AUCEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.92801094F));
		Assert.assertThat(measures.get(MAPEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.44680363F));
		Assert.assertThat(measures.get(MRREvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.6007081F));
		Assert.assertThat(measures.get(NDCGEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.5438155F));
		Assert.assertThat(measures.get(NoveltyEvaluator.class.getSimpleName()), CoreMatchers.equalTo(11.966816F));
		Assert.assertThat(measures.get(PrecisionEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.34595758F));
		Assert.assertThat(measures.get(RecallEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.61634725F));
	}

}
