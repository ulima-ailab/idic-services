import skfuzzy as sk
import numpy as np
from skfuzzy import control as ctrl


list_emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]


def get_fuzzyset_label(component, crisp_val):
    max_val_cat = -1
    max_cat = ''
    for cat in component.terms:
        mf_val = sk.interp_membership(component.universe, component[cat].mf, crisp_val)
        # print(cat + " = " + str(mf_val))
        if max_val_cat < mf_val:
            max_val_cat = mf_val
            max_cat = cat
    return max_cat


class FuzzySystem:

    def __init__(self):
        self.__create_model()
        print("FS created and loaded")

    def __create_model(self):
        # Input 1: Stress level
        ST = np.arange(0, 6, 0.1)
        stress = ctrl.Antecedent(ST, 'stress')
        stress['no_stress'] = sk.sigmf(ST, 1.25, -6)
        stress['mild_stress'] = sk.gaussmf(ST, 3, 0.45)
        stress['high_stress'] = sk.sigmf(ST, 4.75, 5)
        # stress.view()

        # Input 2: Facial emotions
        self.emotions_mf = {'stress': stress}
        for emotion in list_emotions:
            EM = np.arange(0, 1.01, 0.01)
            self.emotions_mf[emotion] = ctrl.Antecedent(EM, emotion)
            self.emotions_mf[emotion][emotion + '_no_present'] = sk.zmf(EM, 0.3, 0.55)
            self.emotions_mf[emotion][emotion + '_present'] = sk.smf(EM, 0.35, 0.6)
            # emotions_mf[emotion].view()

        # Output 1: Level of Persuasion
        LP = np.arange(0, 5, 0.1)
        self.persuasion = ctrl.Consequent(LP, 'persuasion_level')
        self.persuasion['LP0'] = sk.gaussmf(LP, 1, 0.2)
        self.persuasion['LP1'] = sk.gaussmf(LP, 2, 0.2)
        self.persuasion['LP2'] = sk.gaussmf(LP, 3, 0.2)
        self.persuasion['LP3'] = sk.gaussmf(LP, 4, 0.2)
        # persuasion.view()

        rules = []
        rules.append(ctrl.Rule(stress["no_stress"]
                               & self.emotions_mf["angry"]["angry_no_present"]
                               & self.emotions_mf["disgusted"]["disgusted_no_present"]
                               & self.emotions_mf["fearful"]["fearful_no_present"]
                               & self.emotions_mf["sad"]["sad_no_present"], self.persuasion['LP1']))

        rules.append(ctrl.Rule(stress["no_stress"] & self.emotions_mf["angry"]["angry_present"]
                               & self.emotions_mf["disgusted"]["disgusted_no_present"]
                               & self.emotions_mf["fearful"]["fearful_no_present"]
                               & self.emotions_mf["sad"]["sad_no_present"], self.persuasion['LP3']))
        rules.append(ctrl.Rule(stress["no_stress"] & self.emotions_mf["disgusted"]["disgusted_present"]
                               & self.emotions_mf["angry"]["angry_no_present"]
                               & self.emotions_mf["fearful"]["fearful_no_present"]
                               & self.emotions_mf["sad"]["sad_no_present"], self.persuasion['LP3']))
        rules.append(ctrl.Rule(stress["no_stress"] & self.emotions_mf["fearful"]["fearful_present"]
                               & self.emotions_mf["angry"]["angry_no_present"]
                               & self.emotions_mf["disgusted"]["disgusted_no_present"]
                               & self.emotions_mf["sad"]["sad_no_present"], self.persuasion['LP3']))
        rules.append(ctrl.Rule(stress["no_stress"] & self.emotions_mf["sad"]["sad_present"]
                               & self.emotions_mf["angry"]["angry_no_present"]
                               & self.emotions_mf["disgusted"]["disgusted_no_present"]
                               & self.emotions_mf["fearful"]["fearful_no_present"], self.persuasion['LP3']))

        rules.append(ctrl.Rule(stress["mild_stress"]
                               & self.emotions_mf["angry"]["angry_no_present"]
                               & self.emotions_mf["disgusted"]["disgusted_no_present"]
                               & self.emotions_mf["fearful"]["fearful_no_present"]
                               & self.emotions_mf["sad"]["sad_no_present"], self.persuasion['LP0']))

        rules.append(ctrl.Rule(stress["mild_stress"] & self.emotions_mf["angry"]["angry_present"]
                               & self.emotions_mf["disgusted"]["disgusted_no_present"]
                               & self.emotions_mf["fearful"]["fearful_no_present"]
                               & self.emotions_mf["sad"]["sad_no_present"], self.persuasion['LP2']))
        rules.append(ctrl.Rule(stress["mild_stress"] & self.emotions_mf["disgusted"]["disgusted_present"]
                               & self.emotions_mf["angry"]["angry_no_present"]
                               & self.emotions_mf["fearful"]["fearful_no_present"]
                               & self.emotions_mf["sad"]["sad_no_present"], self.persuasion['LP2']))
        rules.append(ctrl.Rule(stress["mild_stress"] & self.emotions_mf["fearful"]["fearful_present"]
                               & self.emotions_mf["angry"]["angry_no_present"]
                               & self.emotions_mf["disgusted"]["disgusted_no_present"]
                               & self.emotions_mf["sad"]["sad_no_present"], self.persuasion['LP2']))
        rules.append(ctrl.Rule(stress["mild_stress"] & self.emotions_mf["sad"]["sad_present"]
                               & self.emotions_mf["angry"]["angry_no_present"]
                               & self.emotions_mf["disgusted"]["disgusted_no_present"]
                               & self.emotions_mf["fearful"]["fearful_no_present"], self.persuasion['LP2']))

        rules.append(ctrl.Rule(stress["high_stress"]
                               & self.emotions_mf["angry"]["angry_no_present"]
                               & self.emotions_mf["disgusted"]["disgusted_no_present"]
                               & self.emotions_mf["fearful"]["fearful_no_present"]
                               & self.emotions_mf["sad"]["sad_no_present"], self.persuasion['LP3']))

        rules.append(ctrl.Rule(stress["high_stress"] & self.emotions_mf["angry"]["angry_present"]
                               & self.emotions_mf["disgusted"]["disgusted_no_present"]
                               & self.emotions_mf["fearful"]["fearful_no_present"]
                               & self.emotions_mf["sad"]["sad_no_present"], self.persuasion['LP2']))
        rules.append(ctrl.Rule(stress["high_stress"] & self.emotions_mf["disgusted"]["disgusted_present"]
                               & self.emotions_mf["angry"]["angry_no_present"]
                               & self.emotions_mf["fearful"]["fearful_no_present"]
                               & self.emotions_mf["sad"]["sad_no_present"], self.persuasion['LP2']))
        rules.append(ctrl.Rule(stress["high_stress"] & self.emotions_mf["fearful"]["fearful_present"]
                               & self.emotions_mf["angry"]["angry_no_present"]
                               & self.emotions_mf["disgusted"]["disgusted_no_present"]
                               & self.emotions_mf["sad"]["sad_no_present"], self.persuasion['LP2']))
        rules.append(ctrl.Rule(stress["high_stress"] & self.emotions_mf["sad"]["sad_present"]
                               & self.emotions_mf["angry"]["angry_no_present"]
                               & self.emotions_mf["disgusted"]["disgusted_no_present"]
                               & self.emotions_mf["fearful"]["fearful_no_present"], self.persuasion['LP2']))

        self.ctrl_sys = ctrl.ControlSystem(rules)

        self.antecedents_labels = []
        for antecedent in self.ctrl_sys.antecedents:
            self.antecedents_labels.append(antecedent.label)

    def process_input(self, input_data):
        print(input_data)
        result = ctrl.ControlSystemSimulation(self.ctrl_sys)
        for key in input_data:
            if key in self.antecedents_labels:
                result.input[key] = input_data[key]
        result.compute()
        result.print_state()

        crisp_val = result.output['persuasion_level']
        print("====== Output: " + str(crisp_val))
        return self.get_output_set(result)

    def get_output_set(self, result):
        max_val_cat = -1
        max_cat = ''
        for c in self.ctrl_sys.consequents:
            for term in c.terms.values():
                mf_val = term.membership_value[result]
                print(term.label, mf_val)
                if max_val_cat <= mf_val:
                    max_val_cat = mf_val
                    max_cat = term.label
        return max_cat


if __name__ == '__main__':
    myFS = FuzzySystem()
    in_data = {"stress": 1, "angry": 0.2, "disgusted": 0.2, "fearful": 0.90, "sad": 0.05, "happy": 0.98}
    out = myFS.process_input(in_data)
    print("========================")
    print(get_fuzzyset_label(myFS.emotions_mf['stress'], in_data["stress"]))
    print(get_fuzzyset_label(myFS.emotions_mf['angry'], in_data["angry"]))
    print(get_fuzzyset_label(myFS.emotions_mf['disgusted'], in_data["disgusted"]))
    print(get_fuzzyset_label(myFS.emotions_mf['fearful'], in_data["fearful"]))
    print(get_fuzzyset_label(myFS.emotions_mf['sad'], in_data["sad"]))

    print(out)
