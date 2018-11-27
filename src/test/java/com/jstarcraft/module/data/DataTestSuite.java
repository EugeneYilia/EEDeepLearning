package com.jstarcraft.module.data;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

import com.jstarcraft.module.data.convertor.ConvertorTestSuite;
import com.jstarcraft.module.data.splitter.SplitterTestSuite;

@RunWith(Suite.class)
@SuiteClasses({

		ConvertorTestSuite.class,

		SplitterTestSuite.class,

		DataSpaceTestCase.class })
public class DataTestSuite {

}
