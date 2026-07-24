import traceback

from agent.ai_consultant import AIConsultant

try:
    consultant = AIConsultant()
    answer = consultant.route('How much inventory should we order for Europe?')
    print('SUCCESS', answer.get('answer_type'))
    print('RECOMMENDATIONS', len(answer.get('recommendations', [])))
except Exception:
    traceback.print_exc()
