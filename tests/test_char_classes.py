"""Character-class composition and native membership-parsing semantics."""
import math
from types import SimpleNamespace
import unittest

import numpy as np

from src.decoder import CharClassRescoringLM, CharClassScorer, prefix_beam_search
from src.recognize import Recognizer


SYMBOLS={2:'a',3:'1'}


class WordLM:
    def start(self):
        return ''

    def advance(self,state,label):
        return state+SYMBOLS[label],.2

    def finish(self,state):
        return .3


class CharClassTests(unittest.TestCase):
    def setUp(self):
        self.weights={'lower':-.99,'number':1.30,'no_char_class':-1.47,'lower_':2.}
        self.classes={'lower':'ab','number':'0123456789','lower_en_us':''}
        self.scorer=CharClassScorer.from_tables(self.classes,self.weights,SYMBOLS)

    def test_explicit_memberships_no_class_fallback_and_unresolved_metadata(self):
        scorer=CharClassScorer.from_tables(self.classes,self.weights,{**SYMBOLS,4:'?',5:'[[space]]'})
        self.assertEqual(dict(scorer.scores),{2:-.99,3:1.30,4:-1.47,5:-1.47})
        self.assertEqual(scorer.empty_classes,('lower_en_us',))
        self.assertEqual(scorer.unmapped_weights,('lower_',))

    def test_class_term_is_once_per_collapsed_token_and_independent_of_word_weight(self):
        logits=np.log(np.array([[.2,.3,.5],[.4,.2,.4],[.3,.3,.4],[.2,.5,.3]]))
        lm=WordLM()
        baseline=dict(prefix_beam_search(logits,SYMBOLS,lm=lm,lm_weight=.7,nbest=1000))
        for sign in (1.,-1.,0.):
            combined=CharClassRescoringLM(lm,self.scorer,word_weight=.7,class_weight=sign)
            actual=dict(prefix_beam_search(logits,SYMBOLS,lm=combined,nbest=1000))
            self.assertEqual(actual.keys(),baseline.keys())
            for text,score in baseline.items():
                expected=score+sign*sum(self.weights['lower' if c=='a' else 'number'] for c in text)
                self.assertAlmostEqual(actual[text],expected,places=12)

    def test_rejection_is_preserved_at_zero_word_weight(self):
        class Reject(WordLM):
            def advance(self,state,label):
                return state,math.inf
            def finish(self,state):
                return math.inf
        combined=CharClassRescoringLM(Reject(),self.scorer,word_weight=0.)
        self.assertEqual(combined.advance('',2),('',math.inf))
        self.assertEqual(combined.finish(''),math.inf)
        class Missing(WordLM):
            def advance(self,state,label):
                return None
        self.assertIsNone(CharClassRescoringLM(Missing(),self.scorer).advance('',2))

    def test_native_membership_semantics_and_nonfinite_weights(self):
        """Native degrades where we used to raise; see decoder.CharClassScorer.from_tables.

        Overlapping membership is last-write-wins (the native code logs "overriding character
        class from ... to ..."), and a class with no weight scores 0.0. 78 of the 361 shipped
        char_class_tables overlap and 163 rely on suffix truncation, so raising on either would
        break most non-English languages.
        """
        # overlap: the LAST class assigned wins, and the override is recorded
        scorer = CharClassScorer.from_tables({'lower': 'a', 'number': 'a'}, self.weights, SYMBOLS)
        a_label = next(l for l, t in SYMBOLS.items() if t == 'a')
        self.assertEqual(scorer.scores[a_label], self.weights['number'])
        self.assertEqual(scorer.overrides, (('a', 'lower', 'number'),))

        # a class with no weight scores 0.0 rather than raising
        scorer = CharClassScorer.from_tables({'unknown': 'a'}, self.weights, SYMBOLS)
        self.assertEqual(scorer.scores[a_label], 0.0)

        # suffixed names truncate to the prefix including the underscore, so a
        # language-specific line feeds the generic class's weight (163 tables rely on this)
        scorer = CharClassScorer.from_tables({'lower_be': 'a'}, self.weights, SYMBOLS)
        self.assertEqual(scorer.scores[a_label], self.weights['lower_'])

        # a bare line contributes nothing at all, leaving its weight inert
        scorer = CharClassScorer.from_tables({'lower_en_us': ''}, self.weights, SYMBOLS)
        self.assertIn('lower_', scorer.unmapped_weights)
        self.assertEqual(scorer.scores[a_label], self.weights['no_char_class'])

        # non-finite weights are still rejected
        with self.assertRaisesRegex(ValueError, 'finite'):
            CharClassScorer.from_tables({'lower': 'a'}, {**self.weights, 'lower': math.nan},
                                        SYMBOLS)

    def test_recognizer_option_does_not_change_greedy(self):
        settings=SimpleNamespace(char_classes=self.classes,
            char_class_weights=[SimpleNamespace(name=k,value=v) for k,v in self.weights.items()])
        mapping=SimpleNamespace(net_to_fst=(2,3,1),net_symbols=('a','1'),blank_index=2)
        r=Recognizer(SimpleNamespace(decoder=settings),None,mapping,None,(),lm=WordLM())
        r.logits_for=lambda strokes:np.log(np.array([[.49,.51,1e-20]]))
        # Acoustically '1' is very slightly ahead, so the class term decides.
        self.assertEqual(r.recognize([],char_class_weight=0.)[0].text,'1')
        # Default is now the binary-faithful POSITIVE weight, under which `number` (+1.30)
        # rewards the digit. The old -1 default rewarded lowercase and produced 'a'.
        self.assertEqual(r.recognize([])[0].text,'1')
        self.assertEqual(r.recognize([],char_class_weight=-1.)[0].text,'a')
        self.assertEqual(r.recognize([],char_class_weight=1.)[0].text,'1')
        # greedy ignores the LM entirely, so the class weight cannot move it
        self.assertEqual(r.recognize([],char_class_weight=-1.,greedy=True)[0].text,'1')
        self.assertIs(r.char_class_scorer,r.char_class_scorer)


if __name__=='__main__':unittest.main()
