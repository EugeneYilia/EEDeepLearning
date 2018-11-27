package com.jstarcraft.module.recommendation.utility;

import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Properties;

import org.apache.commons.lang3.StringUtils;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.jstarcraft.module.recommendation.exception.RecommendationException;
import com.jstarcraft.module.recommendation.recommender.Recommender;

import antlr.RecognitionException;

/**
 * Driver Class Util
 *
 * @author WangYuFeng
 */
@Deprecated
public class DriverUtility {
	/**
	 * driver Class BiMap matches configuration of driver.classes.props
	 */
	private static BiMap<String, String> driverClassBiMap;
	/**
	 * inverse configuration of driver.classes.props
	 */
	private static BiMap<String, String> driverClassInverseBiMap;

	static {
		driverClassBiMap = HashBiMap.create();
		Properties prop = new Properties();
		InputStream is = null;
		try {
			is = DriverUtility.class.getClassLoader().getResourceAsStream("driver.classes.props");
			prop.load(is);
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				is.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		Iterator<Entry<Object, Object>> propIte = prop.entrySet().iterator();
		while (propIte.hasNext()) {
			Entry<Object, Object> entry = propIte.next();
			String key = (String) entry.getKey();
			String value = (String) entry.getValue();
			driverClassBiMap.put(key, value);
		}
		driverClassInverseBiMap = driverClassBiMap.inverse();
	}

	/**
	 * get Class by driver name.
	 *
	 * @param driver
	 *            driver name
	 * @return Class object
	 * @throws ClassNotFoundException
	 *             if can't find the Class
	 */
	public static Class<?> getClass(String driver) {
		try {
			if (StringUtils.isBlank(driver)) {
				return null;
			} else if (StringUtils.contains(driver, ".")) {
				return Class.forName(driver);
			} else {
				String fullName = driverClassBiMap.get(driver);
				return Class.forName(fullName);
			}
		} catch (Exception exception) {
			throw new RecommendationException(exception);
		}
	}

	/**
	 * get Driver Name by clazz
	 *
	 * @param clazz
	 *            clazz name
	 * @return driver name
	 * @throws ClassNotFoundException
	 *             if can't find the Class
	 */
	public static String getDriverName(String clazz) throws ClassNotFoundException {
		if (StringUtils.isBlank(clazz)) {
			return null;
		} else {
			return driverClassInverseBiMap.get(clazz);
		}
	}

	/**
	 * get Driver Name by clazz
	 *
	 * @param clazz
	 *            clazz name
	 * @return driver name
	 * @throws ClassNotFoundException
	 *             if can't find the Class
	 */
	public static String getDriverName(Class<? extends Recommender> clazz) throws ClassNotFoundException {
		if (clazz == null) {
			return null;
		} else {
			String driverName = driverClassInverseBiMap.get(clazz.getName());
			if (StringUtils.isNotBlank(driverName)) {
				return driverName;
			} else {
				return clazz.getSimpleName().toLowerCase().replace("recommender", "");
			}
		}
	}
}
