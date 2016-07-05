package weka.filters;

import java.util.Enumeration;
import java.util.Vector;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Option;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * An instance filter that passes all instances and generate multi-instance
 * instances with given order.
 *
 * @author Qian He <qhe@cs.wpi.edu>
 * @version $Revision$
 */
public class ChainFilter extends Filter {

	/** for serialization */
	private static final long serialVersionUID = 7787365052026028180L;

	/** How many steps to look back in the sequence. */
	protected int m_Order = 1;

	/** Should attributes be included in look back? */
	protected boolean m_IncludeAttributesInOrder = true;

	/**
	 * Returns a string describing this filter
	 *
	 * @return a description of the filter suitable for displaying in the
	 *         explorer/experimenter GUI
	 */
	public String globalInfo() {
		return "An instance filter that passes all instances and generates multi-instance instances with given order.";
	}

	/**
	 * Returns an enumeration describing the available options
	 * 
	 * @return an enumeration of all the available options
	 */
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>(2);

		result.addElement(new Option("\tSet the order of chain.", "N", 1, "-N <order>"));
		result.addElement(new Option("\tSet whether attributes should be included in order.", "F", 0, "-F"));

		return result.elements();
	}

	/**
	 * Parses a given list of options.
	 * 
	 * @param options
	 *            the list of options as an array of strings
	 * @throws Exception
	 *             if an option is not supported
	 */
	@Override
	public void setOptions(String[] options) throws Exception {
		setIncludeAttributesInOrder(Utils.getFlag('F', options));

		String orderString = Utils.getOption('N', options);
		if (orderString.length() != 0) {
			setOrder(Integer.parseInt(orderString));
		} else {
			setOrder(1);
		}

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of the classifier.
	 * 
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();

		result.add("-N");
		result.add("" + getOrder());
		if (getIncludeAttributesInOrder())
			result.add("-F");

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter GUI
	 */
	public String orderTipText() {
		return "The number of steps to look back.";
	}

	/**
	 * @return the m_Order
	 */
	public int getOrder() {
		return m_Order;
	}

	/**
	 * @param order
	 *            the m_Order to set
	 */
	public void setOrder(int order) {
		this.m_Order = order < 1 ? 1 : order;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter GUI
	 */
	public String includeAttributesInOrderTipText() {
		return "If set, attributes will be included in order.";
	}

	/**
	 * @return the m_IncludeAttributesInOrder
	 */
	public boolean getIncludeAttributesInOrder() {
		return m_IncludeAttributesInOrder;
	}

	/**
	 * @param includeAttributesInOrder
	 *            the m_IncludeAttributesInOrder to set
	 */
	public void setIncludeAttributesInOrder(boolean includeAttributesInOrder) {
		this.m_IncludeAttributesInOrder = includeAttributesInOrder;
	}

	/**
	 * Returns the Capabilities of this filter.
	 *
	 * @return the capabilities of this object
	 * @see Capabilities
	 */
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
	 *             if something goes wrong
	 */
	public boolean setInputFormat(Instances instanceInfo) throws Exception {
		super.setInputFormat(instanceInfo);
		return true;
	}

	/**
	 * Input an instance for filtering. Filter requires all training instances
	 * be read before producing output.
	 *
	 * @param instance
	 *            the input instance
	 * @return true if the filtered instance may now be collected with output().
	 * @throws IllegalStateException
	 *             if no input format has been defined.
	 */
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
	 * If the filter requires all instances prior to filtering, output() may now
	 * be called to retrieve the filtered instances.
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
		Instances result = convertInstancesWithOrder(data, getOrder(), getIncludeAttributesInOrder());
		for (int i = 0; i < result.numInstances(); i++)
			push(result.instance(i));

		flushInput();

		m_NewBatch = true;
		return (numPendingOutput() != 0);
	}

	/**
	 * Convert instances to instances with order
	 * 
	 * @param data
	 *            the instances with no order
	 * @param order
	 *            the order with sequence
	 * @param includeAttributes
	 *            whether Attributes have order
	 * 
	 * @return the new instances
	 */
	private Instances convertInstancesWithOrder(Instances data, int order, boolean includeAttributes) {
		// Reconstruct the schema for more Attributes from chain

		// Duplicate (N - 1) times of attributes
		int originalNumAttributes = data.numAttributes();
		int originalClassIndex = data.classIndex() == -1 ? data.numAttributes() - 1 : data.classIndex();
		for (int i = 1; i < order; i++) {
			for (int j = 0; j < originalNumAttributes; j++) {
				if (!includeAttributes && j != originalClassIndex)
					continue;

				Attribute attribute = data.attribute(j);
				data.insertAttributeAt(attribute.copy(attribute.name() + "_" + i), data.numAttributes());
			}
		}

		// Add a "future" version of the class attribute
		Attribute attribute = data.attribute(originalClassIndex);
		data.insertAttributeAt(attribute.copy(attribute.name() + "_future"), data.numAttributes());
		// Set the "future" attribute to be the class attribute
		data.setClass(data.attribute(data.numAttributes() - 1));

		// Change the out format for the filter
		setOutputFormat(data);

		// Fill the new attributes for all instances
		for (int i = order - 1; i < data.numInstances() - 1; i++) {
			Instance instance = data.instance(i);
			Instance futureInstance = data.instance(i + 1);
			// Fill the real label: "future"
			instance.setValue(data.classIndex(), futureInstance.value(originalClassIndex));

			// Fill the attribute of the pass
			for (int j = 1; j < order; j++) {
				Instance previousInstance = data.instance(i - j);

				if (includeAttributes) {
					for (int k = 0; k < originalNumAttributes; k++) {
						instance.setValue(originalNumAttributes * j + k, previousInstance.value(k));
					}
				} else {
					instance.setValue(originalNumAttributes + j - 1, previousInstance.value(originalClassIndex));
				}
			}
		}

		// Remove the first N instances and the last instance
		for (int i = 0; i < order - 1; i++)
			data.remove(0);
		data.remove(data.lastInstance());

		return data;
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision$");
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param argv
	 *            should contain arguments to the filter: use -h for help
	 */
	public static void main(String[] argv) {
		runFilter(new ChainFilter(), argv);
	}
}
