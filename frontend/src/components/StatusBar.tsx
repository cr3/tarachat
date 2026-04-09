import { useTranslation } from 'react-i18next';
import { HealthResponse } from '../types';
import './StatusBar.css';

interface StatusBarProps {
  health: HealthResponse | null;
}

function StatusBar({ health }: StatusBarProps) {
  const { t } = useTranslation();

  if (!health) {
    return (
      <div className="status-bar warning">
        <span className="status-indicator"></span>
        {t('status.connecting')}
      </div>
    );
  }

  const isReady = health.status === 'healthy';

  return (
    <div className={`status-bar ${isReady ? 'ready' : 'initializing'}`}>
      <span className="status-indicator"></span>
      {isReady ? t('status.ready') : t('status.initializing')}
    </div>
  );
}

export default StatusBar;
