package weka.filters.unsupervised.attribute;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Vector;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;

public class CutoffFilter extends Filter {
	/** for serialization */
	private static final long serialVersionUID = 3046357837240892191L;

	/** the cutoff values */
	protected ArrayList<Double> m_Cutoffs = null;

	/** the index of attribute to be discretized */
	protected int m_AttributeIndex = 0;

	/**
	 * whether to treat the cutoff values as the lower bounds or upper bounds
	 */
	protected boolean m_LowerOrUpper = true;

	public CutoffFilter() {
		setCutoffs("0");
	}

	/**
	 * Gets an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>(3);

		result.addElement(
				new Option("\tSpecifies the cutoff values for discretization, separated with comma.", "C", 1, "-C"));

		result.addElement(new Option(
				"\tSpecifies whether to treat the cutoff values as the lower bounds or upper bounds.", "L", 0, "-L"));

		result.addElement(new Option("\tSpecifies the attribute index.", "I", 1, "-I <index>"));

		return result.elements();
	}

	/**
	 * Parses a given list of options.
	 * <p/>
	 * 
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 * 
	 * <pre>
	 *  -C value
	 *  Specifies the cutoff values for discretization, separated with comma.
	 * </pre>
	 * 
	 * <pre>
	 *  -L
	 *  Specifies whether to treat the cutoff values as the lower bounds (true) or upper bounds (false).
	 * </pre>
	 * 
	 * <pre>
	 * -I Specifies the attribute index.
	 * 
	 * <!-- options-end -->
	 * 
	 * @param options
	 *            the list of options as an array of strings
	 * @throws Exception
	 *             if an option is not supported
	 */
	@Override
	public void setOptions(String[] options) throws Exception {
		String cutoffList = Utils.getOption('C', options);
		if (cutoffList.length() > 0) {
			setCutoffs(cutoffList);
		} else {
			setCutoffs("0");
		}

		setLowerOrUpper(Utils.getFlag('L', options));

		String indexString = Utils.getOption("I", options);
		if (indexString.length() > 0) {
			setAttributeIndex(Integer.parseInt(indexString));
		}

		if (getInputFormat() != null) {
			setInputFormat(getInputFormat());
		}

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of the filter.
	 * 
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String[] getOptions() {

		List<String> options = new ArrayList<String>();

		if (getLowerOrUpper())
			options.add("-L");
		options.add("-C");
		options.add(getCutoffs());
		options.add("-I");
		options.add("" + getAttributeIndex());

		return options.toArray(new String[options.size()]);
	}

	/**
	 * Returns the Capabilities of this filter.
	 * 
	 * @return the capabilities of this object
	 * @see Capabilities
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enableAllAttributes();
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enableAllClasses();
		result.enable(Capability.MISSING_CLASS_VALUES);
		result.enable(Capability.NO_CLASS);

		return result;
	}

	/**
	 * Sets the format of the input instances.
	 * 
	 * @param instanceInfo
	 *            an Instances object containing the input instance structure
	 *            (any instances contained in the object are ignored - only the
	 *            structure is required).
	 * @return true if the outputFormat may be collected immediately
	 * @throws Exception
	 *             if the input format can't be set successfully
	 */
	@Override
	public boolean setInputFormat(Instances instanceInfo) throws Exception {
		super.setInputFormat(instanceInfo);

		return false;
	}

	/**
	 * Input an instance for filtering. Ordinarily the instance is processed and
	 * made available for output immediately. Some filters require all instances
	 * be read before producing output.
	 * 
	 * @param instance
	 *            the input instance
	 * @return true if the filtered instance may now be collected with output().
	 * @throws IllegalStateException
	 *             if no input format has been defined.
	 */
	@Override
	public boolean input(Instance instance) {
		if (getInputFormat() == null) {
			throw new IllegalStateException("No input instance format defined");
		}

		if (m_NewBatch) {
			resetQueue();
			m_NewBatch = false;
		}

		bufferInput(instance);
		return false;
	}

	/**
	 * Signifies that this batch of input to the filter is finished. If the
	 * filter requires all instances prior to filtering, output() may now be
	 * called to retrieve the filtered instances.
	 * 
	 * @return true if there are instances pending output
	 * @throws IllegalStateException
	 *             if no input structure has been defined
	 */
	@Override
	public boolean batchFinished() {
		if (getInputFormat() == null) {
			throw new IllegalStateException("No input instance format defined");
		}

		Instances data = getInputFormat();

		// Prepare the new nominal attribute
		ArrayList<String> nominalValues = new ArrayList<>(m_Cutoffs.size() + 2);
		nominalValues.add(convertValueToCutoffRange(Double.NEGATIVE_INFINITY));
		for (int i = 0; i < m_Cutoffs.size() - 1; i++) {
			nominalValues.add(convertValueToCutoffRange((m_Cutoffs.get(i) + m_Cutoffs.get(i + 1)) * 0.5));
		}
		nominalValues.add(convertValueToCutoffRange(Double.POSITIVE_INFINITY));
		Attribute newAttribute = new Attribute("tmp", nominalValues);
		data.insertAttributeAt(newAttribute, m_AttributeIndex + 1);

		// Prepare data for new attribute
		for (int i = 0; i < data.numInstances(); i++) {
			Instance instance = data.instance(i);

			if (!instance.isMissing(m_AttributeIndex)) {
				double value = instance.value(m_AttributeIndex);
				instance.setValue(m_AttributeIndex + 1, convertValueToCutoffRange(value));
			}
		}

		// Change the out format for the filter
		String attributeName = data.attribute(m_AttributeIndex).name();
		if (data.classIndex() == m_AttributeIndex)
			data.setClassIndex(m_AttributeIndex + 1);
		data.deleteAttributeAt(m_AttributeIndex);
		data.renameAttribute(m_AttributeIndex, attributeName);
		setOutputFormat(data);

		// Push data to output
		for (int i = 0; i < data.numInstances(); i++)
			push(data.instance(i));

		flushInput();

		m_NewBatch = true;
		return (numPendingOutput() != 0);
	}

	/**
	 * Returns a string describing this filter
	 * 
	 * @return a description of the filter suitable for displaying in the
	 *         explorer/experimenter GUI
	 */
	public String globalInfo() {
		return "An attribute filter that discretizes a range of numeric attributes in the dataset into nominal attributes with given cutoff values.";
	}

	public String cutoffsTipText() {
		return "The cutoff values for discretization, separated with comma.";
	}

	public String getCutoffs() {
		StringBuffer result = new StringBuffer();

		for (Double cutoff : this.m_Cutoffs) {
			result.append("," + cutoff);
		}

		if (result.length() > 1)
			return result.substring(1);
		else
			return result.toString();
	}

	public void setCutoffs(String cutoffsString) {
		String[] cutoffStrings = cutoffsString.replace(" ", "").split(",");

		this.m_Cutoffs = new ArrayList<>(cutoffStrings.length);
		for (String cutoffString : cutoffStrings) {
			try {
				double cutoff = Double.parseDouble(cutoffString);
				this.m_Cutoffs.add(cutoff);
			} catch (Exception e) {
				System.err.println(e);
			}
		}

		Collections.sort(this.m_Cutoffs);
	}

	public String attributeIndexTipText() {
		return "The index of attribute to be discretized.";
	}

	public int getAttributeIndex() {
		return m_AttributeIndex + 1;
	}

	public void setAttributeIndex(int m_AttributeIndex) {
		this.m_AttributeIndex = m_AttributeIndex - 1;
	}

	public String lowerOrUpperTipText() {
		return "Whether to treat the cutoff values as the lower bounds or upper bounds.";
	}

	public boolean getLowerOrUpper() {
		return m_LowerOrUpper;
	}

	public void setLowerOrUpper(boolean m_LowerOrUpper) {
		this.m_LowerOrUpper = m_LowerOrUpper;
	}

	private String convertValueToCutoffRange(double value) {
		String range = null;

		// Inside cutoff values
		for (int j = 0; j < m_Cutoffs.size() - 1; j++) {
			double left = m_Cutoffs.get(j);
			double right = m_Cutoffs.get(j + 1);

			if (m_LowerOrUpper) {
				if (left <= value && value < right) {
					range = String.format("[%.4f, %.4f)", left, right);
					break;
				}
			} else {
				if (left < value && value <= right) {
					range = String.format("(%.4f, %.4f]", left, right);
					break;
				}
			}
		}
		// Outside cutoff values
		if (m_LowerOrUpper) {
			if (value < m_Cutoffs.get(0)) {
				range = String.format("(-Inf, %.4f)", m_Cutoffs.get(0));
			} else if (m_Cutoffs.get(m_Cutoffs.size() - 1) <= value) {
				range = String.format("[%.4f, +Inf)", m_Cutoffs.get(m_Cutoffs.size() - 1));
			}
		} else {
			if (value <= m_Cutoffs.get(0)) {
				range = String.format("(-Inf, %.4f]", m_Cutoffs.get(0));
			} else if (m_Cutoffs.get(m_Cutoffs.size() - 1) < value) {
				range = String.format("(%.4f, +Inf)", m_Cutoffs.get(m_Cutoffs.size() - 1));
			}
		}

		return range;
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision$");
	}

	/**
	 * Main method for testing this class.
	 * 
	 * @param argv
	 *            should contain arguments to the filter: use -h for help
	 */
	public static void main(String[] args) {
		runFilter(new CutoffFilter(), args);
	}

}
