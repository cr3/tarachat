import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import MessageInput from './MessageInput'

describe('MessageInput', () => {
  it('renders textarea and send button', () => {
    render(<MessageInput onSendMessage={vi.fn()} />)
    expect(
      screen.getByPlaceholderText(/Type your message/),
    ).toBeInTheDocument()
    expect(screen.getByText('Send')).toBeInTheDocument()
  })

  it('send button is disabled when input is empty', () => {
    render(<MessageInput onSendMessage={vi.fn()} />)
    expect(screen.getByText('Send')).toBeDisabled()
  })

  it('send button is disabled when disabled prop is true', () => {
    render(<MessageInput onSendMessage={vi.fn()} disabled />)
    expect(screen.getByText('Send')).toBeDisabled()
  })

  it('textarea is disabled when disabled prop is true', () => {
    render(<MessageInput onSendMessage={vi.fn()} disabled />)
    expect(screen.getByPlaceholderText(/Type your message/)).toBeDisabled()
  })

  it('enables send button when text is entered', async () => {
    const user = userEvent.setup({ delay: null })
    render(<MessageInput onSendMessage={vi.fn()} />)
    await user.type(screen.getByPlaceholderText(/Type your message/), 'hello')
    expect(screen.getByText('Send')).toBeEnabled()
  })

  it('calls onSendMessage and clears input on send click', async () => {
    const user = userEvent.setup({ delay: null })
    const onSend = vi.fn()
    render(<MessageInput onSendMessage={onSend} />)
    const textarea = screen.getByPlaceholderText(/Type your message/)
    await user.type(textarea, 'hello')
    await user.click(screen.getByText('Send'))
    expect(onSend).toHaveBeenCalledWith('hello')
    expect(textarea).toHaveValue('')
  })

  it('sends on Enter key', async () => {
    const user = userEvent.setup({ delay: null })
    const onSend = vi.fn()
    render(<MessageInput onSendMessage={onSend} />)
    await user.type(
      screen.getByPlaceholderText(/Type your message/),
      'hello{Enter}',
    )
    expect(onSend).toHaveBeenCalledWith('hello')
  })

  it('does not send on whitespace-only input', async () => {
    const user = userEvent.setup({ delay: null })
    const onSend = vi.fn()
    render(<MessageInput onSendMessage={onSend} />)
    await user.type(
      screen.getByPlaceholderText(/Type your message/),
      '   {Enter}',
    )
    expect(onSend).not.toHaveBeenCalled()
  })
})
