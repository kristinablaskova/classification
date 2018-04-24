import bayes_samples
import matplotlib.pyplot as plt

#subsetting the eeg values with respect to states
wake_eeg = bayes_samples.data.loc[bayes_samples.data['doctor'] == "Wake"].copy()
nonrem1_eeg = bayes_samples.data.loc[bayes_samples.data['doctor'] == "NonREM1"].copy()
nonrem2_eeg = bayes_samples.data.loc[bayes_samples.data['doctor'] == "NonREM2"].copy()
nonrem3_eeg = bayes_samples.data.loc[bayes_samples.data['doctor'] == "NonREM3"].copy()
rem_eeg = bayes_samples.data.loc[bayes_samples.data['doctor'] == "REM"].copy()
#plotting hists
plt.hist(wake_eeg['eeg'],density=True)
plt.xlabel('EEG values')
plt.ylabel('Probability')
plt.title(r'Distribution of EEG values for Wake state (Emission probability)')
plt.savefig('wake_emission.png')   # save the figure to file
plt.close()

plt.hist(nonrem1_eeg['eeg'],density=True)
plt.xlabel('EEG values')
plt.ylabel('Probability')
plt.title(r'Distribution of EEG values for NonREM1 state (Emission probability)')
plt.savefig('nonrem1_emission.png')   # save the figure to file
plt.close()

plt.hist(nonrem2_eeg['eeg'],density=True)
plt.xlabel('EEG values')
plt.ylabel('Probability')
plt.title(r'Distribution of EEG values for NonREM2 state (Emission probability)')
plt.savefig('nonrem2_emission.png')   # save the figure to file
plt.close()

plt.hist(nonrem3_eeg['eeg'],density=True)
plt.xlabel('EEG values')
plt.ylabel('Probability')
plt.title(r'Distribution of EEG values for NonREM3 state (Emission probability)')
plt.savefig('nonrem3_emission.png')   # save the figure to file
plt.close()

plt.hist(rem_eeg['eeg'],density=True)
plt.xlabel('EEG values')
plt.ylabel('Probability')
plt.title(r'Distribution of EEG values for REM state (Emission probability)')
plt.savefig('rem_emission.png')   # save the figure to file
plt.close()