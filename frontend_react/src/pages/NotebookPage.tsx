import type { FC } from 'react';
import { useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useAppStore } from '../store/useAppStore';
import AppHeader from '../components/Layout/AppHeader';
import MainLayout from '../components/Layout/MainLayout';
import { apiService } from '../services/api';

export const NotebookPage: FC = () => {
    const { id } = useParams<{ id: string }>();
    const setActiveNotebook = useAppStore((s) => s.setActiveNotebook);
    const setSources = useAppStore((s) => s.setSources);
    const setChatMessages = useAppStore((s) => s.setChatMessages);
    const setChatSessionForNotebook = useAppStore((s) => s.setChatSessionForNotebook);
    const setError = useAppStore((s) => s.setError);

    useEffect(() => {
        if (!id) return;

        let mounted = true;

        const loadNotebookData = async () => {
            try {
                setError(null);
                setActiveNotebook(id);
                const [sources, chat] = await Promise.all([
                    apiService.getNotebookSources(id),
                    apiService.getNotebookChatMessages(id),
                ]);
                if (mounted) {
                    setSources(sources);
                    setChatMessages(chat.messages);
                    setChatSessionForNotebook(id, chat.sessionId);
                }
            } catch {
                if (mounted) {
                    setError('Không thể tải nguồn tài liệu của notebook này.');
                    setSources([]);
                    setChatMessages([]);
                    setChatSessionForNotebook(id, null);
                }
            }
        };

        void loadNotebookData();

        return () => {
            mounted = false;
        };
    }, [id, setActiveNotebook, setChatMessages, setChatSessionForNotebook, setError, setSources]);

    if (!id) {
        return null;
    }

    return (
        <div className="flex flex-col h-screen bg-white overflow-hidden">
            {/* TOP APP BAR */}
            <AppHeader />

            {/* 3-COLUMN MAIN LAYOUT */}
            <MainLayout />
        </div>
    );
};

export default NotebookPage;
