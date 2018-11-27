package com.jstarcraft.module.text;

import it.unimi.dsi.fastutil.ints.IntSet;

/**
 * Inverse Document Frequency
 * 
 * @author Birdy
 *
 */
public interface InverseDocumentFrequency {

	IntSet getKeys();

	float getValue(int key);

}
