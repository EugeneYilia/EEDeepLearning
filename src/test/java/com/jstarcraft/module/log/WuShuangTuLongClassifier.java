package com.jstarcraft.module.log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;

import org.apache.commons.io.FileUtils;
import org.junit.Test;

public class WuShuangTuLongClassifier {

	// 将日志按照类型划分到不同的文件.
	@Test
	public void classify() throws Exception {
		Collection<File> files;
		File dataPath = new File("data/wu-shuang-tu-long/logs");
		if (dataPath.isDirectory()) {
			files = FileUtils.listFiles(dataPath, null, true);
		} else {
			files = Arrays.asList(dataPath);
		}

		Collection<String> types = new HashSet<>();

		for (File file : files) {
			System.out.println(file.getName());
			try (FileReader reader = new FileReader(file); BufferedReader in = new BufferedReader(reader)) {
				String line = null;
				while ((line = in.readLine()) != null) {
					String type = line.split("[|]")[0];
					File typeFile = new File("data/wu-shuang-tu-long/games/" + type + ".txt");
					if (types.add(type)) {
						FileUtils.deleteQuietly(typeFile);
						typeFile.createNewFile();
					}
					line = line + '\n';
					FileUtils.write(typeFile, line, true);
				}
			}
		}
	}

}
