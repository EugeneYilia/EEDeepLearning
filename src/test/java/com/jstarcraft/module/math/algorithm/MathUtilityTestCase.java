package com.jstarcraft.module.math.algorithm;

import org.apache.commons.math3.special.Gamma;
import org.junit.Assert;
import org.junit.Test;

import net.sourceforge.jdistlib.math.PolyGamma;

public class MathUtilityTestCase {

	@Test
	public void testGamma() {
		// logGamma遇到负数会变为NaN或者无穷.
		Assert.assertTrue(Double.isNaN(MathUtility.logGamma(0F)) == Double.isNaN(Gamma.logGamma(0D)));
		Assert.assertTrue(Double.isNaN(MathUtility.logGamma(-0F)) == Double.isNaN(Gamma.logGamma(-0D)));
		Assert.assertTrue(Double.isNaN(MathUtility.logGamma(-0.5F)) == Double.isNaN(Gamma.logGamma(-0.5D)));
		Assert.assertTrue(Double.isNaN(MathUtility.logGamma(-1F)) == Double.isNaN(Gamma.logGamma(-1D)));

		Assert.assertTrue(MathUtility.equal(MathUtility.logGamma(0.1F), (float) Gamma.logGamma(0.1D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.logGamma(0.5f), (float) Gamma.logGamma(0.5D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.logGamma(1F), (float) Gamma.logGamma(1D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.logGamma(1.5F), (float) Gamma.logGamma(1.5D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.logGamma(8F), (float) Gamma.logGamma(8D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.logGamma(8.1F), (float) Gamma.logGamma(8.1D)));

		Assert.assertTrue(MathUtility.equal(MathUtility.gamma(0.1F), (float) Gamma.gamma(0.1D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.gamma(0.5F), (float) Gamma.gamma(0.5D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.gamma(1F), (float) Gamma.gamma(1D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.gamma(1.5F), (float) Gamma.gamma(1.5D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.gamma(8F), (float) Gamma.gamma(8D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.gamma(8.1F), (float) Gamma.gamma(8.1D)));
	}

	@Test
	// TODO 通过https://www.wolframalpha.com/input验证,Apache比较准确.
	public void testDigamma() {
		// digamma遇到负整数会变为NaN或者无穷.
		Assert.assertTrue(Double.isNaN(MathUtility.digamma(0F)) == Double.isNaN(Gamma.digamma(0D)));
		Assert.assertTrue(Double.isNaN(MathUtility.digamma(-0F)) == Double.isNaN(Gamma.digamma(-0D)));
		Assert.assertTrue(Double.isNaN(MathUtility.digamma(-1F)) == Double.isNaN(Gamma.digamma(-1D)));

		Assert.assertTrue(MathUtility.equal(MathUtility.digamma(0.1F), (float) Gamma.digamma(0.1D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.inverse(MathUtility.digamma(0.1F), 5), 0.1F));
		Assert.assertTrue(MathUtility.equal(MathUtility.digamma(0.5F), (float) Gamma.digamma(0.5D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.inverse(MathUtility.digamma(0.5F), 5), 0.5F));
		Assert.assertTrue(MathUtility.equal(MathUtility.digamma(1F), (float) Gamma.digamma(1D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.inverse(MathUtility.digamma(1F), 5), 1F));
		Assert.assertTrue(MathUtility.equal(MathUtility.digamma(1.5F), (float) Gamma.digamma(1.5D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.inverse(MathUtility.digamma(1.5F), 5), 1.5F));
		Assert.assertTrue(MathUtility.equal(MathUtility.digamma(8F), (float) Gamma.digamma(8D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.inverse(MathUtility.digamma(8F), 5), 8F));
		Assert.assertTrue(MathUtility.equal(MathUtility.digamma(8.1F), (float) Gamma.digamma(8.1D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.inverse(MathUtility.digamma(8.1F), 5), 8.1F));

		Assert.assertTrue(MathUtility.equal(MathUtility.digamma(-0.1F), (float) Gamma.digamma(-0.1D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.digamma(-0.5F), (float) Gamma.digamma(-0.5D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.digamma(-1.5F), (float) Gamma.digamma(-1.5D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.digamma(-8.1F), (float) Gamma.digamma(-8.1D)));
	}

	@Test
	// TODO 通过https://www.wolframalpha.com/input验证,Apache比较准确.
	public void testTrigamma() {
		// trigamma遇到负整数会变为NaN或者无穷.
		Assert.assertTrue(Double.isNaN(MathUtility.trigamma(0F)) == Double.isNaN(Gamma.trigamma(0D)));
		Assert.assertTrue(Double.isNaN(MathUtility.trigamma(-0F)) == Double.isNaN(Gamma.trigamma(-0D)));
		Assert.assertTrue(Double.isNaN(MathUtility.trigamma(-1F)) == Double.isNaN(Gamma.trigamma(-1D)));

		Assert.assertTrue(MathUtility.equal(MathUtility.trigamma(0.1F), (float) Gamma.trigamma(0.1D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.trigamma(0.5F), (float) Gamma.trigamma(0.5D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.trigamma(1F), (float) Gamma.trigamma(1D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.trigamma(1.5F), (float) Gamma.trigamma(1.5D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.trigamma(8F), (float) Gamma.trigamma(8D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.trigamma(8.1F), (float) Gamma.trigamma(8.1D)));

		Assert.assertTrue(MathUtility.equal(MathUtility.trigamma(-0.1F), (float) Gamma.trigamma(-0.1D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.trigamma(-0.5F), (float) Gamma.trigamma(-0.5D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.trigamma(-1.5F), (float) Gamma.trigamma(-1.5D)));
		Assert.assertTrue(MathUtility.equal(MathUtility.trigamma(-8.1F), (float) Gamma.trigamma(-8.1D)));
	}

	public void test() {
		// log遇到正负零会变为无穷.
		System.out.println(Math.exp(0));
		System.out.println(Math.log(Math.exp(0)));
		System.out.println(Math.log(Math.exp(1)));
		System.out.println(Math.log(Math.exp(2)));
		System.out.println(Math.log(Math.exp(3)));
		System.out.println(Math.log(Math.exp(-3)));

		// psiGamma遇到负整数会变为NaN或者无穷.
		// logGamma遇到负数会变为NaN或者无穷.
		System.out.println(MathUtility.logGamma(1));
		System.out.println(MathUtility.logGamma(2));
		System.out.println(MathUtility.logGamma(3));
		System.out.println(MathUtility.logGamma(-1));
		System.out.println(MathUtility.logGamma(-2));
		System.out.println(MathUtility.logGamma(-3));

		// TODO 准备将PolyGamma与Gamma整合到MathUtility.
		System.out.println(PolyGamma.psigamma(-1.1D, 0));
		System.out.println(PolyGamma.psigamma(-1D, 0));
		System.out.println(PolyGamma.psigamma(-1.1D, 1));
		System.out.println(PolyGamma.psigamma(-1D, 1));
		System.out.println(PolyGamma.psigamma(-1.1D, 2));
		System.out.println(PolyGamma.psigamma(-1D, 2));
	}

}
