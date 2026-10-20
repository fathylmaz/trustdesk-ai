import test from 'node:test';
import assert from 'node:assert/strict';
import { latencyText, parseComparison, readableLabel, scoreText, validateText } from '../src/model.js';

const response = {
  text:'I want to cancel my bank account.',
  predictions:[
    { model:'tfidf', label:'cancel_transfer', score:0.33964, top_predictions:[{label:'cancel_transfer',score:0.33964}],inference_ms:5.959,automatic_routing_enabled:false },
    { model:'distilbert',label:'terminate_account',score:0.65487,top_predictions:[{label:'terminate_account',score:0.65487}],inference_ms:215.353,automatic_routing_enabled:false },
  ],warning:'Experimental local demo only.'
};

test('model comparison parser preserves correct model order and difference',()=>{
  const parsed=parseComparison({...response,predictions:[...response.predictions].reverse()});
  assert.equal(parsed.predictions[0].model,'tfidf');
  assert.equal(parsed.predictions[1].model,'distilbert');
  assert.equal(parsed.agreement,false);
});
test('same intent sets agreement',()=>{
  const parsed=parseComparison({...response,predictions:response.predictions.map(p=>({...p,label:'terminate_account'}))});
  assert.equal(parsed.agreement,true);
});
test('missing model rejected',()=>{
  assert.throws(()=>parseComparison({...response,predictions:[response.predictions[0]]}),/eksik veya geçersiz/);
});
test('backend must explicitly disable auto-routing',()=>{
  assert.throws(()=>parseComparison({...response,predictions:[{...response.predictions[0],automatic_routing_enabled:true},response.predictions[1]]}),/eksik veya geçersiz/);
});
test('text validation',()=>{
  assert.match(validateText('a'),/en az 3/);
  assert.equal(validateText('  valid request '),'');
  assert.match(validateText('x'.repeat(2001)),/2.000/);
});
test('display helpers handle unavailable input without fake numbers',()=>{
  assert.equal(scoreText(0.65487),'0.65487');
  assert.equal(scoreText(undefined),'—');
  assert.equal(latencyText(215.353),'215.35 ms');
  assert.equal(readableLabel('lost_or_stolen_card'),'Kayıp veya çalıntı kart');
});
